"""Displacement-only RAFT training CLI for the ESWG007 tile dataset.

Exists to support the augmentation ablation: train on the 992 originals
(``--originals-only``) vs. the 7,936-tile D4-augmented set, and compare
val loss.  The loss math is ported verbatim from Brock's displacement-only
training loop in ``DICe_RAFT.py`` (pre-strain); strain loss is intentionally
left out tonight.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from raft_model import RAFTStrainModel
from tile_dataset import TileDataset, split_by_tile

LOGGER = logging.getLogger("train_raft")
DEFAULT_TILES_DIR = Path(
    "/home/ren/Documents/Capstone/dice-automation-tools/training_tiles_128_v2"
)


def parse_args() -> argparse.Namespace:
    """Parse training command-line arguments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-dir", type=Path, default=DEFAULT_TILES_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(f"./runs/run_{timestamp}")
    )
    parser.add_argument("--pair-filter", type=str, default=None)
    parser.add_argument("--originals-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--iters", type=int, default=6, help="RAFT GRU iterations at train/val time"
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="DICe-RAFT-Strain")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def configure_logging() -> None:
    """Install a stdout handler with a timestamped format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for repeatable splits and shuffles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def prepare_output_dir(output_dir: Path) -> None:
    """Create an empty output directory or reject a non-empty existing one."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def sequence_loss(
    predictions: list[torch.Tensor],
    gt_d: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """RAFT sequence L1 loss on displacement (Brock's pre-strain formulation)."""
    gamma = 0.8
    n_preds = len(predictions)
    valid_pixels = mask.sum() * 2 + 1e-6
    total_loss: torch.Tensor = torch.zeros((), device=gt_d.device, dtype=gt_d.dtype)
    for i, pred in enumerate(predictions):
        i_weight = gamma ** (n_preds - i - 1)
        i_loss = F.l1_loss(pred * mask, gt_d * mask, reduction="sum") / valid_pixels
        total_loss = total_loss + i_weight * i_loss
    return total_loss


@torch.no_grad()
def validate(
    model: RAFTStrainModel,
    loader: DataLoader,
    device: torch.device,
    iters: int,
    amp_enabled: bool,
) -> float:
    """Masked L1 displacement loss on the final flow only (no sequence weighting)."""
    model.eval()
    val_loss = 0.0
    n_batches = 0
    for batch in loader:
        a = batch["a"].to(device, non_blocking=True)
        b = batch["b"].to(device, non_blocking=True)
        gt_d = batch["disp"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            out = model(a, b, iters=iters)
            loss = (
                F.l1_loss(out["flow"] * mask, gt_d * mask, reduction="sum")
                / (mask.sum() * 2 + 1e-6)
            )
        val_loss += loss.item()
        n_batches += 1
    return val_loss / max(1, n_batches)


def init_wandb(args: argparse.Namespace) -> Optional[Any]:
    """Initialize wandb only when explicitly requested."""
    if not args.wandb:
        return None

    import wandb  # pylint: disable=import-outside-toplevel

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config,
    )


def train(args: argparse.Namespace) -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Run displacement-only RAFT training."""
    set_seed(args.seed)
    prepare_output_dir(args.output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    LOGGER.info("Device: %s", device)

    dataset = TileDataset(
        args.tiles_dir,
        pair_filter=args.pair_filter,
        originals_only=args.originals_only,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No tiles found under {args.tiles_dir} "
            f"(pair_filter={args.pair_filter!r}, originals_only={args.originals_only})."
        )

    train_ds, val_ds = split_by_tile(dataset, args.val_fraction, args.seed)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train and validation splits must both be non-empty")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = RAFTStrainModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    wandb_run = init_wandb(args)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        LOGGER.info("Epoch %d/%d starting", epoch, args.epochs)
        start_time = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            a = batch["a"].to(device, non_blocking=True)
            b = batch["b"].to(device, non_blocking=True)
            gt_d = batch["disp"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(a, b, iters=args.iters)
                total_loss = sequence_loss(out["flows"], gt_d, mask)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(1, n_batches)
        avg_val_loss = validate(model, val_loader, device, args.iters, amp_enabled)
        scheduler.step(avg_val_loss)

        torch.save(model.state_dict(), args.output_dir / "latest.pt")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.output_dir / "best.pt")

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch} train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} lr={current_lr:.4e} "
            f"time={epoch_time:.2f}s",
            flush=True,
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "lr": current_lr,
                    "epoch_time": epoch_time,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    """Entry point."""
    configure_logging()
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
