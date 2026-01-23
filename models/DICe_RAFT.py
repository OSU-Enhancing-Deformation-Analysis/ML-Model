#!/usr/bin/env python3
"""
Train RAFT on image pairs to predict a DICe-style strain map.

Assumptions
-----------
You have paired frames and a ground-truth strain image for each pair.

Default folder layout (you can change args):
dataset_root/
  train/
    a/          # first frame images
    b/          # second frame images
    strain/     # ground-truth strain images (grayscale or RGB)
  val/
    a/
    b/
    strain/

Files are matched by filename (same stem) across a/, b/, strain/.

What this script does
---------------------
1) Uses a pretrained RAFT (torchvision optical flow) as the backbone.
2) Predicts optical flow between frame A and B.
3) Converts flow -> strain map by spatial gradients.
4) Trains with L1 + (optional) SSIM-like smoothness loss against GT strain.
5) Logs train/val losses and sample images to Weights & Biases.

Notes
-----
- Ground-truth strain images are assumed to be already in a comparable scale.
  If your GT is e.g. colored heatmaps, consider converting to scalar strain before training.
- RAFT expects images normalized to [0,1] then standardized to ImageNet stats by default.
- This is a strong baseline; you can improve results by (a) multi-scale strain losses,
  (b) predicting specific components (exx/eyy/exy), (c) better GT scaling/units.

Requirements
------------
pip install torch torchvision pillow wandb matplotlib

Example
-------
python train_raft_strain.py \
  --dataset_root /path/to/dataset_root \
  --project raft-strain \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-4
"""

import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import wandb

# torchvision RAFT (optical flow)
from torchvision.models.optical_flow import raft_small, raft_large
from torchvision.transforms import functional as TF

# -------------------------
# Utils
# -------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_common_stems(dir_a: str, dir_b: str, dir_s: str, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")) -> List[str]:
    def stems_in(d):
        out = set()
        for fn in os.listdir(d):
            if fn.lower().endswith(exts):
                out.add(os.path.splitext(fn)[0])
        return out
    sa = stems_in(dir_a)
    sb = stems_in(dir_b)
    ss = stems_in(dir_s)
    common = sorted(list(sa & sb & ss))
    if len(common) == 0:
        raise RuntimeError(f"No matched triplets found across:\n  {dir_a}\n  {dir_b}\n  {dir_s}")
    return common

def to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    # float32 in [0,1], shape [3,H,W]
    if img.mode != "RGB":
        img = img.convert("RGB")
    t = TF.to_tensor(img)
    return t

def to_tensor_gray(img: Image.Image) -> torch.Tensor:
    # float32 in [0,1], shape [1,H,W]
    if img.mode != "L":
        img = img.convert("L")
    t = TF.to_tensor(img)
    return t

def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    # x: [3,H,W] in [0,1]
    return TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)

def robust_minmax(x: torch.Tensor, q_low=0.01, q_high=0.99, eps=1e-6) -> torch.Tensor:
    # For visualization/logging only: scale per-batch tensor to [0,1] using robust quantiles
    flat = x.flatten()
    lo = torch.quantile(flat, q_low)
    hi = torch.quantile(flat, q_high)
    return ((x - lo) / (hi - lo + eps)).clamp(0, 1)

# -------------------------
# Strain computation
# -------------------------

def finite_difference_gradients(flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    flow: [B,2,H,W]  (u,v) displacement in pixels
    returns du/dx, du/dy, dv/dx, dv/dy each [B,1,H,W]
    Central differences with replicate padding.
    """
    u = flow[:, 0:1]  # [B,1,H,W]
    v = flow[:, 1:2]

    # Pad (left, right, top, bottom)
    u_pad = F.pad(u, (1, 1, 1, 1), mode="replicate")
    v_pad = F.pad(v, (1, 1, 1, 1), mode="replicate")

    # Central differences
    du_dx = (u_pad[:, :, 1:-1, 2:] - u_pad[:, :, 1:-1, :-2]) * 0.5
    du_dy = (u_pad[:, :, 2:, 1:-1] - u_pad[:, :, :-2, 1:-1]) * 0.5
    dv_dx = (v_pad[:, :, 1:-1, 2:] - v_pad[:, :, 1:-1, :-2]) * 0.5
    dv_dy = (v_pad[:, :, 2:, 1:-1] - v_pad[:, :, :-2, 1:-1]) * 0.5

    return du_dx, du_dy, dv_dx, dv_dy

def flow_to_strain_scalar(flow: torch.Tensor, mode: str = "equivalent") -> torch.Tensor:
    """
    Convert flow to a scalar strain map.

    Small-strain tensor:
      exx = du/dx
      eyy = dv/dy
      exy = 0.5*(du/dy + dv/dx)

    Returns:
      strain: [B,1,H,W]
    mode:
      - "equivalent": sqrt(exx^2 + eyy^2 + 2*exy^2)  (common scalar magnitude)
      - "exx", "eyy", "exy"
    """
    du_dx, du_dy, dv_dx, dv_dy = finite_difference_gradients(flow)
    exx = du_dx
    eyy = dv_dy
    exy = 0.5 * (du_dy + dv_dx)

    if mode == "exx":
        return exx
    if mode == "eyy":
        return eyy
    if mode == "exy":
        return exy

    # "equivalent" strain magnitude (one reasonable scalar)
    strain = torch.sqrt(exx * exx + eyy * eyy + 2.0 * exy * exy + 1e-8)
    return strain

# -------------------------
# Dataset
# -------------------------

class PairStrainDataset(Dataset):
    def __init__(self, root: str, split: str, resize: Tuple[int, int] = None, strain_as_gray: bool = True):
        """
        root: dataset_root
        split: "train" or "val"
        resize: (H,W) or None
        strain_as_gray: if True, converts GT strain to grayscale scalar map [1,H,W]
        """
        self.dir_a = os.path.join(root, split, "a")
        self.dir_b = os.path.join(root, split, "b")
        self.dir_s = os.path.join(root, split, "strain")
        for d in (self.dir_a, self.dir_b, self.dir_s):
            if not os.path.isdir(d):
                raise RuntimeError(f"Missing directory: {d}")

        self.stems = list_common_stems(self.dir_a, self.dir_b, self.dir_s)
        self.resize = resize
        self.strain_as_gray = strain_as_gray

        # Pick extension per file by scanning dir_s when loading; keep stems only.

    def __len__(self):
        return len(self.stems)

    def _open_by_stem(self, directory: str, stem: str) -> Image.Image:
        # Find a file with any common extension
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            path = os.path.join(directory, stem + ext)
            if os.path.exists(path):
                return Image.open(path)
        raise FileNotFoundError(f"Could not find file for stem '{stem}' in {directory}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.stems[idx]
        img_a = self._open_by_stem(self.dir_a, stem)
        img_b = self._open_by_stem(self.dir_b, stem)
        gt_s  = self._open_by_stem(self.dir_s, stem)

        if self.resize is not None:
            # PIL expects (W,H)
            w, h = self.resize[1], self.resize[0]
            img_a = img_a.resize((w, h), resample=Image.BILINEAR)
            img_b = img_b.resize((w, h), resample=Image.BILINEAR)
            gt_s  = gt_s.resize((w, h), resample=Image.BILINEAR)

        a = normalize_imagenet(to_tensor_rgb(img_a))
        b = normalize_imagenet(to_tensor_rgb(img_b))

        if self.strain_as_gray:
            s = to_tensor_gray(gt_s)  # [1,H,W] in [0,1]
        else:
            s = to_tensor_rgb(gt_s)   # [3,H,W] in [0,1] (not recommended unless your GT is already scalar)

        return {"a": a, "b": b, "strain": s, "stem": stem}

# -------------------------
# Model wrapper
# -------------------------

class RAFTStrainModel(nn.Module):
    def __init__(self, raft_variant: str = "small", pretrained: bool = True, strain_mode: str = "equivalent"):
        super().__init__()
        self.strain_mode = strain_mode
        if raft_variant == "large":
            self.raft = raft_large(pretrained=pretrained, progress=True)
        else:
            self.raft = raft_small(pretrained=pretrained, progress=True)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        a,b: [B,3,H,W] normalized
        Returns dict with:
          flow:   [B,2,H,W]
          strain: [B,1,H,W]
        """
        # torchvision RAFT returns a list of flow predictions (coarse->fine), last is best
        flows = self.raft(a, b)
        if isinstance(flows, (list, tuple)):
            flow = flows[-1]
        else:
            flow = flows

        strain = flow_to_strain_scalar(flow, mode=self.strain_mode)
        return {"flow": flow, "strain": strain}

# -------------------------
# Losses / Metrics
# -------------------------

def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)

def smoothness_loss(flow: torch.Tensor, weight: float = 0.0) -> torch.Tensor:
    """
    Encourage smooth flow -> smoother strain maps.
    """
    if weight <= 0:
        return flow.new_tensor(0.0)
    du_dx, du_dy, dv_dx, dv_dy = finite_difference_gradients(flow)
    loss = charbonnier(du_dx).mean() + charbonnier(du_dy).mean() + charbonnier(dv_dx).mean() + charbonnier(dv_dy).mean()
    return weight * loss

def strain_l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # pred: [B,1,H,W], gt: [B,1,H,W]
    return F.l1_loss(pred, gt)

def strain_rmse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(pred, gt) + 1e-8)

# -------------------------
# Train / Eval
# -------------------------

@dataclass
class TrainConfig:
    dataset_root: str
    project: str
    run_name: str
    raft_variant: str
    pretrained: bool
    strain_mode: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    smooth_weight: float
    resize_h: int
    resize_w: int
    num_workers: int
    log_images_every: int
    device: str
    amp: bool
    seed: int
    out_dir: str

def make_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    resize = (cfg.resize_h, cfg.resize_w) if (cfg.resize_h > 0 and cfg.resize_w > 0) else None
    train_ds = PairStrainDataset(cfg.dataset_root, "train", resize=resize, strain_as_gray=True)
    val_ds   = PairStrainDataset(cfg.dataset_root, "val",   resize=resize, strain_as_gray=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainConfig) -> Dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_rmse = 0.0
    total = 0

    for batch in loader:
        a = batch["a"].to(cfg.device, non_blocking=True)
        b = batch["b"].to(cfg.device, non_blocking=True)
        gt = batch["strain"].to(cfg.device, non_blocking=True)

        out = model(a, b)
        pred = out["strain"]

        l1 = strain_l1_loss(pred, gt).item()
        rmse = strain_rmse(pred, gt).item()

        bs = a.size(0)
        total_l1 += l1 * bs
        total_rmse += rmse * bs
        total += bs

    return {
        "val/strain_l1": total_l1 / max(total, 1),
        "val/strain_rmse": total_rmse / max(total, 1),
    }

def log_examples_to_wandb(batch, out, step: int, prefix: str = "val"):
    # Log a few images: A, B, GT strain, Pred strain
    # Convert normalized A/B back to [0,1] for display (approx)
    a = batch["a"].detach().cpu()
    b = batch["b"].detach().cpu()
    gt = batch["strain"].detach().cpu()
    pred = out["strain"].detach().cpu()

    # Unnormalize imagenet for logging
    mean = torch.tensor(IMAGENET_MEAN).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(1,3,1,1)
    a_vis = (a * std + mean).clamp(0,1)
    b_vis = (b * std + mean).clamp(0,1)

    gt_vis = robust_minmax(gt)
    pr_vis = robust_minmax(pred)

    n = min(4, a_vis.shape[0])
    images = []
    for i in range(n):
        images.append(wandb.Image(a_vis[i].permute(1,2,0).numpy(), caption=f"{prefix} A"))
        images.append(wandb.Image(b_vis[i].permute(1,2,0).numpy(), caption=f"{prefix} B"))
        images.append(wandb.Image(gt_vis[i,0].numpy(), caption=f"{prefix} GT strain"))
        images.append(wandb.Image(pr_vis[i,0].numpy(), caption=f"{prefix} Pred strain"))

    wandb.log({f"{prefix}/examples": images}, step=step)

def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_loader, val_loader = make_loaders(cfg)

    model = RAFTStrainModel(
        raft_variant=cfg.raft_variant,
        pretrained=cfg.pretrained,
        strain_mode=cfg.strain_mode
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device.startswith("cuda")))

    # W&B init
    wandb.init(
        project=cfg.project,
        name=cfg.run_name if cfg.run_name else None,
        config=cfg.__dict__,
    )

    global_step = 0
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for it, batch in enumerate(train_loader, start=1):
            a = batch["a"].to(cfg.device, non_blocking=True)
            b = batch["b"].to(cfg.device, non_blocking=True)
            gt = batch["strain"].to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and cfg.device.startswith("cuda"))):
                out = model(a, b)
                pred = out["strain"]
                flow = out["flow"]

                l_strain = strain_l1_loss(pred, gt)
                l_smooth = smoothness_loss(flow, cfg.smooth_weight)
                loss = l_strain + l_smooth

            scaler.scale(loss).backward()

            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # Log training metrics
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/strain_l1": l_strain.item(),
                    "train/flow_smooth": l_smooth.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                },
                step=global_step
            )

            global_step += 1

        # Validation
        val_metrics = evaluate(model, val_loader, cfg)
        wandb.log(val_metrics, step=global_step)

        # Log a few examples (use first batch from val loader)
        if (epoch % cfg.log_images_every) == 0:
            model.eval()
            with torch.no_grad():
                batch_val = next(iter(val_loader))
                a = batch_val["a"].to(cfg.device, non_blocking=True)
                b = batch_val["b"].to(cfg.device, non_blocking=True)
                out = model(a, b)
                log_examples_to_wandb(batch_val, out, step=global_step, prefix="val")

        # Checkpoint
        val_loss = val_metrics["val/strain_l1"]
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg.__dict__,
            "val_strain_l1": val_loss,
        }
        last_path = os.path.join(cfg.out_dir, "last.pt")
        torch.save(ckpt, last_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(cfg.out_dir, "best.pt")
            torch.save(ckpt, best_path)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{cfg.epochs} | train_loss={epoch_loss/len(train_loader):.5f} "
              f"| val_strain_l1={val_loss:.5f} | time={dt:.1f}s")

    wandb.finish()

# -------------------------
# Inference helper (optional)
# -------------------------

@torch.no_grad()
def infer_strain_map(model_path: str, img_a_path: str, img_b_path: str, out_png: str,
                     raft_variant: str = "small", strain_mode: str = "equivalent",
                     resize: Tuple[int,int] = None, device: str = "cuda"):
    """
    Loads a checkpoint and writes a visualization PNG of predicted strain (0..255 scaled robustly).
    """
    device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = RAFTStrainModel(raft_variant=raft_variant, pretrained=False, strain_mode=strain_mode).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    img_a = Image.open(img_a_path).convert("RGB")
    img_b = Image.open(img_b_path).convert("RGB")
    if resize is not None:
        w, h = resize[1], resize[0]
        img_a = img_a.resize((w,h), resample=Image.BILINEAR)
        img_b = img_b.resize((w,h), resample=Image.BILINEAR)

    a = normalize_imagenet(to_tensor_rgb(img_a)).unsqueeze(0).to(device)
    b = normalize_imagenet(to_tensor_rgb(img_b)).unsqueeze(0).to(device)

    out = model(a, b)
    strain = out["strain"].squeeze(0).squeeze(0).detach().cpu()  # [H,W]
    strain_vis = (robust_minmax(strain) * 255.0).byte().numpy()

    Image.fromarray(strain_vis, mode="L").save(out_png)
    print(f"Saved predicted strain visualization to: {out_png}")

# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--project", type=str, default="raft-strain")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--out_dir", type=str, default="./checkpoints_raft_strain")

    p.add_argument("--raft_variant", type=str, default="small", choices=["small", "large"])
    p.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained RAFT weights")

    p.add_argument("--strain_mode", type=str, default="equivalent", choices=["equivalent", "exx", "eyy", "exy"])

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--smooth_weight", type=float, default=0.01)

    p.add_argument("--resize_h", type=int, default=384, help="Set <=0 to disable resizing")
    p.add_argument("--resize_w", type=int, default=512, help="Set <=0 to disable resizing")

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_images_every", type=int, default=1)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")

    p.add_argument("--seed", type=int, default=42)

    # Optional inference mode
    p.add_argument("--infer", action="store_true")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--img_a", type=str, default="")
    p.add_argument("--img_b", type=str, default="")
    p.add_argument("--out_png", type=str, default="pred_strain.png")

    return p.parse_args()

def main():
    args = parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    cfg = TrainConfig(
        dataset_root=args.dataset_root,
        project=args.project,
        run_name=args.run_name,
        raft_variant=args.raft_variant,
        pretrained=args.pretrained,
        strain_mode=args.strain_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        smooth_weight=args.smooth_weight,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        num_workers=args.num_workers,
        log_images_every=args.log_images_every,
        device=device,
        amp=args.amp,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    if args.infer:
        if not args.ckpt or not args.img_a or not args.img_b:
            raise ValueError("--infer requires --ckpt --img_a --img_b")
        resize = (cfg.resize_h, cfg.resize_w) if (cfg.resize_h > 0 and cfg.resize_w > 0) else None
        infer_strain_map(
            model_path=args.ckpt,
            img_a_path=args.img_a,
            img_b_path=args.img_b,
            out_png=args.out_png,
            raft_variant=cfg.raft_variant,
            strain_mode=cfg.strain_mode,
            resize=resize,
            device=cfg.device,
        )
        return

    train(cfg)

if __name__ == "__main__":
    main()
