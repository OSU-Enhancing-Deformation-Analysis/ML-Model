# %% [markdown]
#
# Important things to know about this notebook:
#
# - Model architecture is RAFT
# - It contains code to tile images
# - There are additional config options for validation and tiling
# - Vector fields have been tuned to have little to no averaging bias
# - Perlin noise has been changed to Simplex and is using a new library
# - Fields can be randomly flipped
# - Fields are not stored in tuples but (2, H, W) arrays
# - Contains to generate visuals of the fields using flow_vis
# - Fields applied to images use `order=1` interpolation
# - Reduced the number of shape masks
# - All shapes are now scaled to 0-1
# - All shapes have been drastically sped up
# - Contains code to visualize shapes
# - New synthetic dataset using and IterableDataset
# - 75% have shapes masks, of those 50% are inverted, 50% are blured
# - 50% have inverted inside, 50% have newfield inside
# - Uses AdamW for the optimizer
#

# %% MARK: Imports
import os
import glob
import re
import random
import sys
import time
from typing import Callable, List, Tuple, Dict, TypeAlias, Any

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import math
from PIL import Image
from collections import defaultdict

# Monkey patch for pyfastnoisesimd
if not hasattr(np, "product"):
    np.product = np.prod  # type: ignore
import pyfastnoisesimd as fns

from dataclasses import dataclass
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.ndimage import rotate, shift
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon
from functools import wraps
import flow_vis
import argparse
from argparse import Namespace

import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn.functional as F

# %% MARK: Constants
a = False
assert "This file is not ready for use yet" and a

# Load the CUDA GPU if available
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
GPU = torch.cuda.get_device_name(0)
print(f"Using {GPU} GPU with {GPU_MEMORY} GB of memory")

# ( GB - 0.5 (buffer)) / 0.65 = BATCH_SIZE
BATCH_SIZE = int((GPU_MEMORY - 1.5) / 0.55) - 3
NUM_WORKERS = 0

# Run save frequency for saving snapshots & checkpoints
# This also controls how often the example images are saved
# Measured in hours
SNAPSHOT_FREQUENCY = 0.5
# Evaluation Frequency
# This is a percentage of the training set
# 1 means after every training image
# 0.1 means after ever 1/10th of the training set
# Values over 1 will be interpreted as # of batches
EVALUATION_FREQUENCY = 1

# Enable Weights and Biases Logging
# Disable this if running locally or just testing
WANDB_ENABLED = True

# Model name for saving files and in wandb
RUN_NAME = "b5-unknown-combo-test"

# %% MARK: Tile Loading

# Relative path to the folder of images
# This folder is recursively searched so make sure it
# doesn't contain a symbolic link to itself
IMAGES_DIR = "../../raw/g79/"
IMAGES_FILE_EXTENSION = ".tif"
# True if the folder contains tiles, False if it contains full images
DIR_CONTAINS_TILES = False
# Relative path to the folder of testing images
# These get evaluated using the model and logged durring training
# The folder should contain images names "image_a.png", "image_b.png", "test_a.png", "test_b.png", etc.
# Importantly the "_a" and "_b" are required and serve as the two inputs to the model
EXAMPLE_IMAGES_DIR = "../../raw/g79test/"
EXAMPLE_IMAGES_FILE_EXTENSION = ".png"

TILE_SIZE = 256  # Pixels
MAX_TILES = 1000
# Validation split
# The validation split is the percentage of the dataset that is used for validation
VALIDATION_SPLIT = 0.05

# -- Tiling options for images not yet tiled --
# Size of the overlap between tiles
# A larger number means more tiles
OVERLAP_SIZE = 64  # Pixels
# Will produce tiles with black regions outside the original image
# The CENTER_SIZE will always have image content
INCLUDE_OUTSIDE = False

# %% MARK: Load CLI Options

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_name",
    default=RUN_NAME,
    help="Name of the run used for Weights and Biases and file names",
)
parser.add_argument(
    "--batch_size_multiplier",
    default=1,
    type=float,
    help="Multiply the batch size by this number",
)
parser.add_argument(
    "--evaluation_frequency",
    default=EVALUATION_FREQUENCY,
    type=float,
    help="How often to evaluate the model as a percentage of the training set. Values over 1 are interpreted as # of batches",
)
parser.add_argument(
    "--snapshot_frequency",
    default=SNAPSHOT_FREQUENCY,
    type=float,
    help="How often to save a snapshot of the model measured in hours",
)
parser.add_argument(
    "--validation_split",
    default=VALIDATION_SPLIT,
    type=float,
    help="The percentage of the dataset to use for validation",
)
parser.add_argument(
    "--max_tiles",
    default=MAX_TILES,
    type=int,
    help="The maximum number of tiles to generate",
)
parser.add_argument(
    "--images_dir",
    default=IMAGES_DIR,
    help="The directory containing the images to use for training",
)
parser.add_argument(
    "--example_images_dir",
    default=EXAMPLE_IMAGES_DIR,
    help="The directory containing the images to use for example generation",
)
parser.add_argument(
    "--images_file_extension",
    default=IMAGES_FILE_EXTENSION,
    help="The file extension of the images to use",
)
parser.add_argument(
    "--example_images_file_extension",
    default=EXAMPLE_IMAGES_FILE_EXTENSION,
    help="The file extension of the images to use",
)
parser.add_argument(
    "--tile_size", default=TILE_SIZE, type=int, help="The size of the tiles to use"
)
parser.add_argument(
    "--overlap_size",
    default=OVERLAP_SIZE,
    type=int,
    help="The size of the overlap between tiles",
)
parser.add_argument(
    "--num_workers",
    default=NUM_WORKERS,
    type=int,
    help="The number of workers to use for the dataloader",
)
# Boolean handling

wandb_arg_group = parser.add_mutually_exclusive_group()
wandb_arg_group.add_argument(
    "--wandb_enabled",
    dest="wandb",
    action="store_true",
    help="Enable Weights and Biases logging",
)
wandb_arg_group.add_argument(
    "--wandb_disabled",
    dest="wandb",
    action="store_false",
    help="Disable Weights and Biases logging",
)
parser.set_defaults(wandb=WANDB_ENABLED)

dircontainstiles_arg_group = parser.add_mutually_exclusive_group()
dircontainstiles_arg_group.add_argument(
    "--dir_contains_tiles",
    dest="dir_contains_tiles",
    action="store_true",
    help="Specify that the images directory contains tiles",
)
dircontainstiles_arg_group.add_argument(
    "--dir_contains_full_images",
    dest="dir_contains_tiles",
    action="store_false",
    help="Specify that the images directory contains full images",
)
parser.set_defaults(dir_contains_tiles=DIR_CONTAINS_TILES)

includeoutside_arg_group = parser.add_mutually_exclusive_group()
includeoutside_arg_group.add_argument(
    "--include_outside",
    dest="include_outside",
    action="store_true",
    help="Whether to include the outside of the image in the tiles",
)
includeoutside_arg_group.add_argument(
    "--exclude_outside",
    dest="include_outside",
    action="store_false",
    help="Whether to exclude the outside of the image in the tiles",
)
parser.set_defaults(include_outside=INCLUDE_OUTSIDE)


args, _unknown_args = parser.parse_known_args()

print(" === Command Line Arguments ===\n")
print(args)
print("\n\n")

RUN_NAME = args.run_name
WANDB_ENABLED = args.wandb
EVALUATION_FREQUENCY = args.evaluation_frequency
SNAPSHOT_FREQUENCY = args.snapshot_frequency
VALIDATION_SPLIT = args.validation_split
MAX_TILES = args.max_tiles
TILE_SIZE = args.tile_size
OVERLAP_SIZE = args.overlap_size
INCLUDE_OUTSIDE = args.include_outside
IMAGES_DIR = args.images_dir
EXAMPLE_IMAGES_DIR = args.example_images_dir
IMAGES_FILE_EXTENSION = args.images_file_extension
EXAMPLE_IMAGES_FILE_EXTENSION = args.example_images_file_extension
DIR_CONTAINS_TILES = args.dir_contains_tiles
NUM_WORKERS = args.num_workers
BATCH_SIZE = int(BATCH_SIZE * args.batch_size_multiplier)

SNAPSHOT_FILE = f"{RUN_NAME}.pth"

# %% MARK: Load Tiles
print(" === Image Loading ===\n")
print(f"- Loading images from {IMAGES_DIR}")
IMAGE_PATHS = glob.glob(
    os.path.join(IMAGES_DIR, "**/*" + IMAGES_FILE_EXTENSION), recursive=True
)
print(f"- Found {len(IMAGE_PATHS)} images")

# Prepare the images and load into memory
SHIFT_SIZE = TILE_SIZE - OVERLAP_SIZE


def create_tiles(image: np.ndarray) -> List[npt.NDArray[np.uint8]]:
    """Split image into overlapping tiles.

    Args:
        image (np.ndarray): The image to split into tiles. Must be shape (height, width)

    Returns:
        List[Tile]: A list of Tile objects, each containing the data and position of a tile in the image.

    """
    height, width = image.shape
    tiles = []

    if INCLUDE_OUTSIDE:
        for y in range(-TILE_SIZE // 2, height - TILE_SIZE // 2, SHIFT_SIZE):
            for x in range(-TILE_SIZE // 2, width - TILE_SIZE // 2, SHIFT_SIZE):

                # Calculate tile boundaries
                y_start = max(y, 0)
                x_start = max(x, 0)
                y_end = min(y + TILE_SIZE, height)
                x_end = min(x + TILE_SIZE, width)

                shift_x = x_start - x
                shift_y = y_start - y

                # Extract tile data
                tile_data = image[y_start:y_end, x_start:x_end]

                # Pad if necessary
                if tile_data.shape != (TILE_SIZE, TILE_SIZE):
                    padded_data = np.zeros(
                        (TILE_SIZE, TILE_SIZE), dtype=tile_data.dtype
                    )
                    padded_data[
                        shift_y : shift_y + tile_data.shape[0],
                        shift_x : shift_x + tile_data.shape[1],
                    ] = tile_data
                    tile_data = padded_data

                tiles.append(tile_data)
    else:
        for y in range(0, height - OVERLAP_SIZE, SHIFT_SIZE):
            for x in range(0, width - OVERLAP_SIZE, SHIFT_SIZE):

                # Calculate tile boundaries
                y_start = max(y, 0)
                x_start = max(x, 0)
                y_end = min(y + TILE_SIZE, height)
                x_end = min(x + TILE_SIZE, width)

                if y + TILE_SIZE > height:
                    # Move tile up to fit on image

                    y_start = height - TILE_SIZE
                    y_end = height
                    y = y_start

                elif x + TILE_SIZE > width:
                    # Move tile left to fit on image

                    x_start = width - TILE_SIZE
                    x_end = width
                    x = x_start

                shift_x = x_start - x
                shift_y = y_start - y

                # Extract tile data
                tile_data = image[y_start:y_end, x_start:x_end]

                # Pad if necessary
                if tile_data.shape != (TILE_SIZE, TILE_SIZE):
                    padded_data = np.zeros(
                        (TILE_SIZE, TILE_SIZE), dtype=tile_data.dtype
                    )
                    padded_data[
                        shift_y : shift_y + tile_data.shape[0],
                        shift_x : shift_x + tile_data.shape[1],
                    ] = tile_data
                    tile_data = padded_data

                tiles.append(tile_data)

    return tiles


def create_tile_references(image_idx: int) -> List[Tuple[int, int, int]]:
    """Split image into overlapping tiles.

    Args:
        image_idx (int): The index of the image to split into tiles.

    Returns:
        List[Tuple[int, int, int]]: A list of Tile references, each containing the image index, x, and y position of a tile in the image.

    """
    image_path = IMAGE_PATHS[image_idx]
    image = Image.open(image_path)
    width, height = image.size
    tiles = []

    if INCLUDE_OUTSIDE:
        for y in range(-TILE_SIZE // 2, height - TILE_SIZE // 2, SHIFT_SIZE):
            for x in range(-TILE_SIZE // 2, width - TILE_SIZE // 2, SHIFT_SIZE):
                tiles.append((image_idx, x, y))
    else:
        for y in range(0, height - OVERLAP_SIZE, SHIFT_SIZE):
            for x in range(0, width - OVERLAP_SIZE, SHIFT_SIZE):
                if y + TILE_SIZE > height:
                    y = height - TILE_SIZE
                elif x + TILE_SIZE > width:
                    x = width - TILE_SIZE

                tiles.append((image_idx, x, y))

    return tiles


def get_tile_from_reference(reference: Tuple[int, int, int]) -> np.ndarray:
    """Get a tile from a reference (image index, x, y).

    Args:
        reference (Tuple[int, int, int]): The reference to the tile (image index, x, y).

    Returns:
        np.ndarray: The tile at the reference.
    """
    image_idx, x, y = reference
    img_path = IMAGE_PATHS[image_idx]
    image = Image.open(img_path).convert("L")  # Convert to grayscale values (0-255)
    height, width = image.size
    image = np.array(image)

    y_start = max(y, 0)
    x_start = max(x, 0)
    y_end = min(y + TILE_SIZE, height)
    x_end = min(x + TILE_SIZE, width)

    shift_x = x_start - x
    shift_y = y_start - y

    tile_data = image[y_start:y_end, x_start:x_end]

    tile_data = np.pad(
        tile_data,
        (
            (shift_y, TILE_SIZE - tile_data.shape[0] - shift_y),
            (shift_x, TILE_SIZE - tile_data.shape[1] - shift_x),
        ),
        mode="constant",
        constant_values=0,
    )

    return tile_data.astype(np.float32, copy=False)


TILES: List[Tuple[int, int, int]]

if DIR_CONTAINS_TILES:
    # Loading tiles into memory
    TILES = [(idx, 0, 0) for idx in range(len(IMAGE_PATHS))]
else:
    # Cut the images into tiles
    print("- Cutting images into tiles")
    TILES = []
    for idx in range(len(IMAGE_PATHS)):
        TILES.extend(create_tile_references(idx))


print(f"- Totaling {len(TILES)} tiles")
NUM_TILES = min(MAX_TILES, len(TILES))
print(f"- Limiting to {NUM_TILES} tiles")
TILES = TILES[:NUM_TILES]

if NUM_TILES == 0:
    print("No tiles found. Exiting.")
    exit(1)

NUM_TRAINING_TILES = int(NUM_TILES * (1 - VALIDATION_SPLIT))
NUM_VALIDATION_TILES = int(NUM_TILES * VALIDATION_SPLIT)
print(
    f"\n= {NUM_TRAINING_TILES} training tiles\n= {NUM_VALIDATION_TILES} validation tiles\n"
)

if EVALUATION_FREQUENCY > 1:
    EVALUATION_FREQUENCY = int(EVALUATION_FREQUENCY)
else:
    EVALUATION_FREQUENCY = int(NUM_TRAINING_TILES / BATCH_SIZE * EVALUATION_FREQUENCY)

## Read the Examples images and tile them
example_pairs = defaultdict(dict)

print(f"- Loading example images from {EXAMPLE_IMAGES_DIR}")
for filename in os.listdir(EXAMPLE_IMAGES_DIR):
    if filename.endswith(EXAMPLE_IMAGES_FILE_EXTENSION):
        parts = filename.rsplit("_", maxsplit=1)
        pair_id = parts[0]
        suffix = parts[1].split(".")[0]  # 'a' or 'b'
        example_pairs[pair_id][suffix] = filename


def _load_image_as_tensor(path: str) -> np.ndarray:
    path = os.path.join(EXAMPLE_IMAGES_DIR, path)
    image = Image.open(path)
    img = 2 * (np.array(image).astype(np.float32) / 255.0) - 1
    return np.stack([img, img, img])


# Convert to list of tuples, sorted by pair ID
EXAMPLE_TILES: List[Tuple[str, Tuple[Tensor, Tensor]]] = [
    (
        pair["a"],
        (
            torch.from_numpy(_load_image_as_tensor(pair["a"])),
            torch.from_numpy(_load_image_as_tensor(pair["b"])),
        ),
    )
    for key, pair in sorted(example_pairs.items())
    if "a" in pair and "b" in pair
]

print(f"- Found {len(EXAMPLE_TILES)} example tile pairs")
print("\n\n")

# %% MARK: Vector Fields

farray: TypeAlias = npt.NDArray[np.float32]

VECTOR_FIELDS: Dict[
    str,
    Callable[
        [farray, farray],
        Tuple[farray, farray],
    ],
] = {}


def vector_field():
    def decorator(func: Callable):
        field_name = func.__name__
        VECTOR_FIELDS[field_name] = func

        @wraps(func)
        def wrapper(X: farray, Y: farray) -> Tuple[farray, farray]:
            dX, dY = func(X, Y)
            return dX.astype(np.float32, copy=False), dY.astype(np.float32, copy=False)

        return wrapper

    return decorator


@vector_field()
def translation_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return np.ones_like(X), np.zeros_like(Y)


@vector_field()
def rotation_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return -Y / 2.0, X / 2.0


@vector_field()
def shear_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return np.ones_like(X) / 2.0, X / 2.0


@vector_field()
def shear_field2(X, Y):
    return Y / 1.5, np.zeros_like(Y)


@vector_field()
def scale_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return X / 2.0, Y / 2.0


@vector_field()
def gradient_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return X**2 / 2.0, Y**2 / 2.0


@vector_field()
def gradient_field2(X: farray, Y: farray) -> Tuple[farray, farray]:
    field = X**2 + Y**2
    return np.gradient(field, axis=0) * 15, np.gradient(field, axis=1) * 15


@vector_field()
def harmonic_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    return np.sin(X), np.cos(Y)


@vector_field()
def harmonic_field2(X: farray, Y: farray) -> Tuple[farray, farray]:
    innerX = 0.75 * np.pi * X
    innerY = 0.75 * np.pi * Y
    sinX: farray = np.sin(innerX)
    cosX: farray = np.cos(innerX)
    sinY: farray = np.sin(innerY)
    cosY: farray = np.cos(innerY)
    return (sinX * cosY) * 1.1, (-cosX * sinY) * 1.1


@vector_field()
def pearling_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    epsilon = 0.1  # Small smoothing factor
    r = np.sqrt(X**2 + Y**2 + epsilon**2)  # Smoothed radius
    dx = np.sin(2 * np.pi * X) * X / r
    dy = np.cos(2 * np.pi * Y) * Y / r
    return dx * 1.1, dy * 1.1


@vector_field()
def vortex_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    r = np.sqrt(X**2 + Y**2)
    return -Y / (r**2 + 0.1), X / (r**2 + 0.1)


@vector_field()
def perlin_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    noise = fns.Noise(seed=np.random.randint(2**31))
    noise.frequency = random.uniform(0.01, 0.03)
    noise.noiseType = fns.NoiseType.Simplex
    noise_x = noise.genAsGrid((X.shape[0], X.shape[1]))
    noise.seed = np.random.randint(2**31)
    noise_y = noise.genAsGrid((X.shape[0], X.shape[1]))

    return noise_x, noise_y


@vector_field()
def swirl_field(X: farray, Y: farray) -> Tuple[farray, farray]:
    epsilon = 0.1  # Small smoothing factor
    radius = np.sqrt(X**2 + Y**2 + epsilon**2)  # Smoothed radius
    angle = np.arctan2(Y, X)

    magnitude = np.tanh(radius)  # Scales velocity smoothly to 0 at origin
    cosX: farray = np.cos(angle + radius)
    sinX: farray = np.sin(angle + radius)
    dx = cosX * magnitude
    dy = sinX * magnitude

    return dx, dy


@vector_field()
def vortex_field2(X: farray, Y: farray) -> Tuple[farray, farray]:
    radius = np.sqrt(X**2 + Y**2)

    dx = -Y / (radius + 0.1)
    dy = X / (radius + 0.1)
    return dx * 1.1, dy * 1.1


@dataclass
class VectorField:
    name: str
    field_func: Callable[[farray, farray], Tuple[farray, farray]]
    amplitude: float = 1.0
    center: Tuple[float, float] = (0, 0)
    scale: float = 1.0
    rotation: float = 0.0

    def randomize(self) -> None:
        self.center = (random.random() * 3 - 1.5, random.random() * 3 - 1.5)
        self.scale = random.uniform(0.8, 2.0)
        self.rotation = random.random() * 2 * np.pi
        self.amplitude = random.uniform(-1, 1)
        self.flip_x = random.choice([1, -1])
        self.flip_y = random.choice([1, -1])

    def apply(self, X: farray, Y: farray) -> Tuple[farray, farray]:
        X_scaled = X * self.scale
        Y_scaled = Y * self.scale

        X_centered = X_scaled - self.center[0]
        Y_centered = Y_scaled - self.center[1]

        cos_theta = np.cos(self.rotation)
        sin_theta = np.sin(self.rotation)

        X_rot_pos = X_centered * cos_theta - Y_centered * sin_theta
        Y_rot_pos = X_centered * sin_theta + Y_centered * cos_theta

        dx, dy = self.field_func(X_rot_pos, Y_rot_pos)

        X_rot = dx * cos_theta + dy * sin_theta
        Y_rot = -dx * sin_theta + dy * cos_theta

        X_rot = X_rot * self.flip_x
        Y_rot = Y_rot * self.flip_y

        return X_rot * self.amplitude, Y_rot * self.amplitude


class VectorFieldComposer:
    def __init__(self):
        self.fields: List[VectorField] = []

        self.grid_X, self.grid_Y = np.meshgrid(
            np.linspace(-1, 1, TILE_SIZE), np.linspace(-1, 1, TILE_SIZE)
        )

        self.pos_grid = np.array(
            np.meshgrid(np.arange(TILE_SIZE), np.arange(TILE_SIZE))
        ).astype(np.float32)

    def add_field(self, field_type: str, randomize: bool = True, **kwargs) -> None:
        if field_type not in VECTOR_FIELDS:
            raise ValueError(f"Unknown field type: {field_type}")

        field = VectorField(
            name=field_type, field_func=VECTOR_FIELDS[field_type], **kwargs
        )
        if randomize:
            field.randomize()
        self.fields.append(field)

    def clear(self):
        self.fields.clear()

    def pop_field(self) -> None:
        self.fields.pop()

    def last(self) -> VectorField:
        return self.fields[-1]

    def compute_combined_field(self) -> farray:
        """
        Computes the combined field by summing the individual fields.

        Returns:
            farray: The combined field. (H, W, 2) array.
        """
        total = np.zeros((2, TILE_SIZE, TILE_SIZE), dtype=np.float32)

        for field in self.fields:
            dx, dy = field.apply(self.grid_X, self.grid_Y)
            total[0] += dx
            total[1] += dy

        return total

    def apply_to_image(self, image: farray, field: farray | None = None) -> farray:
        """
        Applies a field to an image.

        Parameters:
            image (farray): The image to apply the field to. Must be a float array.
            field (farray): The field to apply to the image. Should be the same size as the field (H, W, 2).
        """
        if field is None:
            field = self.compute_combined_field()

        new_pos = self.pos_grid - field

        warped_image = map_coordinates(
            image,
            [new_pos[1], new_pos[0]],
            order=1,
            mode="wrap",
        )

        return warped_image.astype(np.float32)


print(" === Vector Fields ===\n")
print("- Vector Fields:")
for field_type in VECTOR_FIELDS.keys():
    print(f"  - {field_type}")
print("\n")

# %% MARK: Flow Visualizations


def flow_to_color(flow_uv: farray):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2] or [2,H,W]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    if flow_uv.shape[0] == 2:
        flow_uv = np.transpose(flow_uv, (1, 2, 0))

    return flow_vis.flow_to_color(flow_uv)


def flow_to_RG_color(flow_uv, clip_flow=None):
    """
    Creates a red-green visualization from displacement data.

    Args:
        flow_uv (np.ndarray): Flow in UV format of shape (H, W, 2).
        clip_flow (float, optional): Clip maximum flow value. Defaults to None.

    Returns:
        np.ndarray: Red-green flow visualization of shape (H, W, 3).
    """

    assert flow_uv.ndim == 3, "input flow must have three dimensions"

    if flow_uv.shape[0] == 2:
        flow_uv = np.transpose(flow_uv, (1, 2, 0))

    assert flow_uv.shape[2] == 2, "input flow must have shape (H, W, 2)"

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad + epsilon)
    v = v / (rad + epsilon)

    u = (u + 1) / 2
    v = (v + 1) / 2

    flow_img = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    flow_img[:, :, 0] = np.floor(255 * u)
    flow_img[:, :, 1] = np.floor(255 * v)
    flow_img[:, :, 2] = np.floor(255 * 0.5)

    return flow_img


def _visualise_vector_field(vector_field, save=True):
    print(f"Visualising {vector_field}")
    composer = VectorFieldComposer()

    fig, axes = plt.subplots(
        3, 8, gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 3, 1]}, figsize=(12, 5)
    )
    left_3_cols = axes[:, :3]
    right_3_cols = axes[:, 3:6]

    for ax in left_3_cols.flatten():
        ax.axis("off")
        composer.clear()
        composer.add_field(vector_field)
        field = composer.compute_combined_field()
        ax.imshow(flow_to_color(field))

    for ax in right_3_cols.flatten():
        ax.axis("off")
        composer.clear()
        composer.add_field(vector_field)
        field = composer.compute_combined_field()

        ax.imshow(flow_to_RG_color(field))

    samples = 10000
    total_min = 0
    total_max = 0
    total = 0
    min_min = 1000
    max_min = -1000
    min_max = 1000
    max_max = -1000

    min_time = 100000
    max_time = -100000
    total_time = 0

    all_dx = np.zeros((samples, TILE_SIZE, TILE_SIZE))
    all_dy = np.zeros((samples, TILE_SIZE, TILE_SIZE))
    all_mag = np.zeros((samples, TILE_SIZE, TILE_SIZE))
    for i in range(samples):
        start_time = time.time()
        composer.clear()
        composer.add_field(vector_field)
        dx, dy = composer.compute_combined_field()

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        total_time += elapsed_time
        min_time = min(min_time, elapsed_time)
        max_time = max(max_time, elapsed_time)

        magnitude = np.sqrt(dx**2 + dy**2)

        min_mag = np.min(magnitude)
        max_mag = np.max(magnitude)
        avg_mag = np.mean(magnitude)

        total_min += min_mag
        total_max += max_mag
        total += avg_mag

        min_min = min(min_min, min_mag)
        max_min = max(max_min, min_mag)
        min_max = min(min_max, max_mag)
        max_max = max(max_max, max_mag)

        all_dx[i] = dx
        all_dy[i] = dy
        all_mag[i] = magnitude

    avg_min = total_min / samples
    avg_max = total_max / samples
    avg_avg = total / samples
    time_avg = (total_time - min_time - max_time) / (samples - 2)

    print(
        f"Min Time: {min_time:.6f}, Max Time: {max_time:.6f}, Avg Time: {time_avg:.6f} (ms)"
    )
    print("Magnitude:")
    print(f"Lowest Min {min_min:.2f}, Highest Min {max_min:.2f}, Avg {avg_min:.2f}")
    print(f"Lowest Max {min_max:.2f}, Highest Max {max_max:.2f}, Avg {avg_max:.2f}")
    print(f"Avg: {avg_avg}, 1/x avg=0.5: {avg_avg * 2}")

    axes[0, 6].set_title("DX")
    axes[0, 6].hist(all_dx.ravel(), bins=100)
    axes[1, 6].set_title("DY")
    axes[1, 6].hist(all_dy.ravel(), bins=100)
    axes[2, 6].set_title("Magnitude")
    axes[2, 6].hist(all_mag.ravel(), bins=100)

    avg_dx = np.mean(all_dx, axis=0)
    avg_dy = np.mean(all_dy, axis=0)
    avg_mag = np.mean(all_mag, axis=0)

    axes[0, 7].set_title("DX")
    pos = axes[0, 7].imshow(avg_dx, cmap="viridis", vmin=-1, vmax=1)
    fig.colorbar(pos, ax=axes[0, 7])
    axes[1, 7].set_title("DY")
    pos = axes[1, 7].imshow(avg_dy, cmap="viridis", vmin=-1, vmax=1)
    fig.colorbar(pos, ax=axes[1, 7])
    axes[2, 7].set_title("Magnitude")
    pos = axes[2, 7].imshow(avg_mag, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(pos, ax=axes[2, 7])

    if save:
        os.makedirs(f"testing_visuals/vector_fields", exist_ok=True)
        plt.savefig(f"testing_visuals/vector_fields/{vector_field}-norm.png")

    return fig


def _visualise_all_vector_fields(save=True):
    for vector_field in VECTOR_FIELDS.keys():
        _visualise_vector_field(vector_field, save=save)


# %% MARK: Shape Masks
# All shape masks should return values from 0-1


def create_square_shape(size: int) -> farray:
    """Creates a square shape: centered -> rotated -> translated."""

    half_size = size // 2

    # 1. Centered square size
    scale = random.uniform(0.1, 1.0) * size
    square_size = int(scale)

    # Create an empty array
    square_array = np.zeros((size, size), dtype=np.uint8)

    # 2. Draw square at the center
    start = half_size - square_size // 2
    end = start + square_size
    square_array[start:end, start:end] = 1

    # 3. Rotate around center
    rotation = random.uniform(0, 2 * np.pi)
    if rotation != 0:
        square_array = rotate(
            square_array, np.degrees(rotation), order=0, reshape=False
        )

    # 4. Translate
    max_offset = half_size + square_size // 2
    offset_x = random.randint(-max_offset, max_offset)
    offset_y = random.randint(-max_offset, max_offset)
    square_array = shift(square_array, shift=(offset_y, offset_x), order=0)

    return square_array.astype(np.float32, copy=False)


def create_circle_shape(size: int) -> farray:
    """Creates a circle shape"""
    half_size = size // 2

    # 1. Random radius (scaled)
    radius = random.uniform(0.1, 0.5) * size

    # 2. Create grid for circle mask
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - half_size) ** 2 + (Y - half_size) ** 2)
    circle_mask = dist_from_center <= radius

    # 3. Convert to binary image
    circle_array = np.zeros((size, size), dtype=np.uint8)
    circle_array[circle_mask] = 1

    # 4. Translate (within bounds to avoid clipping)
    max_offset = half_size + int(radius)
    offset_x = random.randint(-max_offset, max_offset)
    offset_y = random.randint(-max_offset, max_offset)
    circle_array = shift(circle_array, shift=(offset_y, offset_x), order=0)

    return circle_array.astype(np.float32, copy=False)


def create_blob_shape(size: int) -> farray:
    """Creates a soft-edged (feathered) circle using a Gaussian falloff."""
    half_size = size // 2

    # 1. Random radius
    radius = random.uniform(0.1, 0.5) * size
    sigma = radius / 2.5  # Controls feather softness

    # 2. Create distance grid
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - half_size) ** 2 + (Y - half_size) ** 2)

    # 3. Apply Gaussian falloff
    circle_array = np.exp(-(dist**2) / (2 * sigma**2))

    # Normalize to range [0, 1] (optional)
    circle_array = circle_array / circle_array.max()

    # 4. Translate (feathered falloff works with shift too)
    max_offset = half_size + int(radius)
    offset_x = random.randint(-max_offset, max_offset)
    offset_y = random.randint(-max_offset, max_offset)
    circle_array = shift(
        circle_array, shift=(offset_y, offset_x), order=1, mode="constant", cval=0.0
    )

    return circle_array.astype(np.float32, copy=False)


def create_gradient_shape(size: int) -> farray:
    """
    Creates a size×size uint8 array with a repeating 1D gradient,
    randomly translated, scaled, and rotated.
    """
    # random parameters (use numpy for speed)
    pos = np.random.uniform(-size / 2, size / 2, size=2).astype(np.float32)
    scale = np.float32(np.random.uniform(0.1, 1.5))
    theta = np.float32(np.random.uniform(0, 2 * np.pi))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # generate coordinate grids
    Y, X = np.indices((size, size), dtype=np.float32)

    # apply translation and scale
    x_rel = (X + pos[0]) / scale
    y_rel = (Y + pos[1]) / scale

    # rotate coordinates (only the x‐component matters for a 1D gradient)
    x_rot = x_rel * cos_t - y_rel * sin_t

    # make it repeat every `size` units and normalize to [0..255]
    # adding `size` before mod ensures positivity
    grad = np.mod(x_rot + size, size) / size

    return grad.astype(np.float32, copy=False)


def create_checkerboard_shape(size: int) -> farray:
    """Creates a black-and-white checkerboard pattern."""
    num_checkers = random.randint(1, 12)

    checker_size = size // num_checkers

    # 1. Create one tile (2x2 alternating pattern)
    tile = np.array([[1, 0], [0, 1]], dtype=np.float32)
    checker_addition = math.ceil(num_checkers / 2)
    reps = (num_checkers // 2 + checker_addition, num_checkers // 2 + checker_addition)
    checkerboard = np.tile(tile, reps)

    # 2. Scale it up to full resolution
    checkerboard = np.kron(
        checkerboard, np.ones((checker_size, checker_size), dtype=np.float32)
    )

    # 3. Random rotation
    angle_deg = random.uniform(0, 360)
    checkerboard = rotate(
        checkerboard, angle=angle_deg, reshape=False, order=1, mode="constant", cval=0.0
    )

    # 4. Random translation (within limits to avoid full clipping)
    max_offset = checker_size // 2
    dx = random.randint(-max_offset, max_offset)
    dy = random.randint(-max_offset, max_offset)
    checkerboard = shift(
        checkerboard, shift=(dy, dx), order=1, mode="constant", cval=0.0
    )

    width, height = checkerboard.shape

    # Center crop
    start_x = max(0, width // 2 - size // 2)
    end_x = min(width, start_x + size)

    start_y = max(0, height // 2 - size // 2)
    end_y = min(height, start_y + size)

    checkerboard = checkerboard[start_y:end_y, start_x:end_x]
    # 5. Normalize to [0, 1]
    checkerboard = checkerboard / checkerboard.max()

    return checkerboard.astype(np.float32, copy=False)


def create_rectangle_shape(size: int) -> farray:
    """Creates a rectangle shape: centered -> rotated -> translated."""

    half_size = size // 2

    # 1. Centered rectangle size
    square_size_x = int(random.uniform(0.1, 1.0) * size)
    square_size_y = int(random.uniform(0.1, 1.0) * size)

    # Create an empty array
    square_array = np.zeros((size, size), dtype=np.uint8)

    # 2. Draw square at the center
    start_x = half_size - square_size_x // 2
    end_x = start_x + square_size_x
    start_y = half_size - square_size_y // 2
    end_y = start_y + square_size_y
    square_array[start_y:end_y, start_x:end_x] = 1

    # 3. Rotate around center
    rotation = random.uniform(0, 2 * np.pi)
    if rotation != 0:
        square_array = rotate(
            square_array, np.degrees(rotation), order=0, reshape=False
        )

    # 4. Translate
    max_offset_x = half_size + square_size_x // 2
    max_offset_y = half_size + square_size_y // 2
    offset_x = random.randint(-max_offset_x, max_offset_x)
    offset_y = random.randint(-max_offset_y, max_offset_y)
    square_array = shift(square_array, shift=(offset_y, offset_x), order=0)

    return square_array.astype(np.float32, copy=False)


def create_ellipse_shape(size: int) -> farray:
    """Creates an ellipse shape with randomized position, scale, and rotation."""
    half_size = size // 2

    semi_major_axis_a = random.uniform(size * 0.1, size * 0.48)
    semi_major_axis_b = random.uniform(size * 0.1, size * 0.48)

    angle = random.uniform(0, 2 * math.pi)

    Y, X = np.ogrid[:size, :size]
    x_coords = X - half_size
    y_coords = Y - half_size

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    x_ellipse_frame = x_coords * cos_a - y_coords * sin_a
    y_ellipse_frame = x_coords * sin_a + y_coords * cos_a

    ellipse_mask = (x_ellipse_frame / semi_major_axis_a) ** 2 + (
        y_ellipse_frame / semi_major_axis_b
    ) ** 2 <= 1

    ellipse_array = np.zeros((size, size), dtype=np.uint8)
    ellipse_array[ellipse_mask] = 1

    max_offset = half_size + int(semi_major_axis_a)
    offset_x = random.randint(-max_offset, max_offset)
    offset_y = random.randint(-max_offset, max_offset)
    ellipse_array = shift(
        ellipse_array, shift=(offset_y, offset_x), order=1, mode="constant", cval=0.0
    )

    return ellipse_array.astype(np.float32, copy=False)


def create_polygon_shape(size: int) -> farray:
    """Creates a regular polygon shape with randomized position, scale, and rotation."""
    half_size = size // 2

    num_sides = random.randint(3, 12)

    max_pos = int(size * 0.6)
    position_x = random.randint(-max_pos, max_pos)
    position_y = random.randint(-max_pos, max_pos)

    scale = random.uniform(0.1, 1.5) * size

    angles = np.sort(np.random.rand(num_sides) * 2 * np.pi)
    radii = scale * (0.5 + np.random.rand(num_sides) * 0.5)

    x = radii * np.cos(angles) + half_size + position_x
    y = radii * np.sin(angles) + half_size + position_y

    rr, cc = polygon(x, y)
    polygon_array = np.zeros((size, size), dtype=np.uint8)
    rr = np.clip(rr, 0, size - 1)
    cc = np.clip(cc, 0, size - 1)
    polygon_array[rr, cc] = 1

    return polygon_array.astype(np.float32, copy=False)


def create_perlin_noise_shape(
    size: int,
) -> farray:
    """Creates a Perlin noise pattern with randomized parameters, position, scale, and rotation (scale/position effect)."""
    seed = np.random.randint(2**31)

    perlin = fns.Noise(seed=seed)
    perlin.frequency = random.uniform(0.01, 0.03)
    perlin.noiseType = fns.NoiseType.Simplex
    perlin.perturb.perturbType = fns.PerturbType.NoPerturb
    result = perlin.genAsGrid((size, size))

    return result.astype(np.float32, copy=False)


def create_stripes_shape(size: int) -> farray:
    """Creates a stripes pattern with randomized stripe width, angle, position, and scale (scale/position effect)."""
    pos = np.random.uniform(-size / 2, size / 2, size=2).astype(np.float32)
    scale = np.float32(np.random.uniform(1.0, 3.0))
    theta = np.float32(np.random.uniform(0, 2 * np.pi))
    stripe_width = 20
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Create coordinate grids
    Y, X = np.indices((size, size), dtype=np.float32)

    # Translate and scale
    x_rel = (X + pos[0]) / scale
    y_rel = (Y + pos[1]) / scale

    # Rotate – only x-axis needed for stripe alignment
    x_rot = x_rel * cos_t - y_rel * sin_t

    # Determine stripe position – use floor division
    stripe_idx = (np.floor(x_rot / stripe_width).astype(int)) % 2

    return stripe_idx.astype(np.float32, copy=False)


def create_voronoi_pattern(size: int) -> farray:
    """
    Generates a Voronoi cell pattern as a grayscale image.
    Each region is assigned a unique value.

    Parameters:
        size       – output image will be size×size
        num_points – number of Voronoi seed points

    Returns:
        uint8 NumPy array of shape (size, size)
    """
    num_points = random.randint(10, 30)

    # Step 1: Place random seed points
    points = np.random.randint(-size // 2, size // 2, size=(num_points, 2))

    # Step 2: Create a label image where each point gets a unique ID
    label_image = np.full((size, size), fill_value=-1, dtype=np.int32)
    for i, (y, x) in enumerate(points):
        label_image[y, x] = i

    # Step 3: Compute distance to nearest seed and assign labels
    output = distance_transform_edt(label_image == -1, return_indices=True)
    if not output:
        return np.zeros((size, size), dtype=np.float32)

    distances, labels = output
    final_labels = label_image[labels[0], labels[1]]

    # Step 4: Normalize to grayscale (0–255)
    max_label = np.max(final_labels)
    normalized = final_labels.astype(np.float32) / (max_label + 1)
    return normalized.astype(np.float32, copy=False)


# --- Updated Shape Function List with Variations ---
shape_functions: List[Callable[[int], farray]] = [
    create_square_shape,
    create_circle_shape,
    create_blob_shape,
    create_gradient_shape,
    create_checkerboard_shape,
    create_rectangle_shape,
    create_ellipse_shape,
    create_polygon_shape,
    create_perlin_noise_shape,
    create_stripes_shape,
    create_voronoi_pattern,
]


print(" === Shape Functions ===\n")
print("- Shape Functions:")
for shape_function in shape_functions:
    print(f"  - {shape_function.__name__}")
print("\n")

# %% MARK: Shape Visualizations


def _visualise_shape(shape_function, save=True):
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(3, 7, figure=fig, wspace=0.4, hspace=0.4)

    index = 0
    for row in range(3):
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(shape_function(TILE_SIZE), cmap="gray")
            ax.axis("off")
            index += 1

    samples = 1000

    all = np.zeros((samples, TILE_SIZE, TILE_SIZE))
    total_time = 0
    for i in range(samples):
        start_time = time.time()
        shape_function(TILE_SIZE)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        total_time += elapsed_time

        all[i] = shape_function(TILE_SIZE)

    print(f"Average Time: {total_time / samples:.6f} ms")

    avg = np.mean(all, axis=0)

    ax_large = fig.add_subplot(gs[:, 3:5])
    pos = ax_large.imshow(avg, cmap="viridis")
    ax_large.set_title("Average")
    ax_large.axis("off")
    fig.colorbar(pos, ax=ax_large)

    ax_large2 = fig.add_subplot(gs[:, 5:7])
    pos = ax_large2.imshow(avg, cmap="viridis", vmin=0, vmax=1)
    ax_large2.set_title("Average")
    ax_large2.axis("off")
    fig.colorbar(pos, ax=ax_large2)

    if save:
        os.makedirs(f"testing_visuals/shape_masks", exist_ok=True)
        plt.savefig(f"testing_visuals/shape_masks/{shape_function.__name__}-norm.png")

    return fig


def _visualise_all_shapes(save=True):
    for shape_function in shape_functions:
        print(shape_function.__name__)
        _visualise_shape(shape_function, save=save)


# %% MARK: Dataset


class SyntheticDataset(IterableDataset[Tuple[farray, farray, farray]]):
    VERSION = "v5-raft"

    def __init__(self, validation=False):
        self.validation = validation

        self.composer = VectorFieldComposer()
        self.available_fields = list(VECTOR_FIELDS.keys())

        self.pos_grid = np.stack(
            np.meshgrid(np.arange(TILE_SIZE), np.arange(TILE_SIZE)), axis=-1
        )  # (H, W, 2)

        self.loop_random = random.Random(int(time.time()))

    def __iter__(self):
        if self.validation:
            for i in range(NUM_VALIDATION_TILES):
                idx = (
                    self.loop_random.randint(0, NUM_VALIDATION_TILES - 1)
                    + NUM_TRAINING_TILES
                )
                seed = self.loop_random.randint(0, 2**31)
                yield self._generate(idx, seed)
        else:
            while True:
                idx = self.loop_random.randint(0, NUM_TRAINING_TILES - 1)
                seed = self.loop_random.randint(0, 2**31)
                yield self._generate(idx, seed)

    def _generate(self, idx: int, seed: int, scale: float = 1.0):

        # Seed the random number generator
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 1. Generate a random field
        self.composer.clear()

        num_fields = random.randint(1, 2)  # 1 or 2 fields to stack
        for _ in range(num_fields):
            field_type = random.choice(self.available_fields)
            self.composer.add_field(field_type, randomize=True)

        computed_field = np.array(self.composer.compute_combined_field())

        # 2. Generate a random shape with a 75% chance
        self.composer.clear()

        chance_of_shape = random.random()
        if chance_of_shape > 0.75:
            final_field = computed_field

        else:
            # Pick a random shape function for the mask
            shape_function = random.choice(shape_functions)
            shape_layer: farray = shape_function(
                TILE_SIZE
            )  # Returns and image TileSize x TileSize of floats 0-1

            # Random chance to invert the mask
            if random.random() > 0.5:
                shape_layer = 1 - shape_layer  # Invert mask

            # Random chance to blur the mask
            if random.random() > 0.5:
                sigma = random.uniform(0.5, 2.0)
                shape_layer = gaussian_filter(shape_layer, sigma=sigma).astype(
                    np.float32, copy=False
                )

            # Morph the image with a new field
            self.composer.clear()

            num_fields = random.randint(1, 2)  # 1 or 2 fields to stack
            for _ in range(num_fields):
                field_type = random.choice(self.available_fields)
                self.composer.add_field(field_type, randomize=True)

            # Apply the morph to the mask
            mask = self.composer.apply_to_image(shape_layer)

            # Randomly pick between using the mask to invert the original field or applying another field
            # to the inner region
            field_a = computed_field
            field_b = -computed_field

            if random.random() > 0.5:
                # Create another field to put in the inner region
                self.composer.clear()

                num_fields = random.randint(1, 2)  # 1 or 2 fields to stack
                for _ in range(num_fields):
                    field_type = random.choice(self.available_fields)
                    self.composer.add_field(field_type, randomize=True)

                field_b = self.composer.compute_combined_field()

            # Apply the mask to the fields
            # mask = mask
            final_field: farray = field_a * mask + field_b * (1 - mask)

        tile_reference = TILES[idx]
        image = get_tile_from_reference(tile_reference)

        image: farray = 2 * (image / 255.0) - 1.0

        # Randomly transform the image
        # Rotate 0, 90, 180, 270
        image = np.rot90(image, random.randint(0, 3))

        # Flip horizontally, vertically
        if random.random() > 0.5:
            image = np.flip(image, random.randint(0, 1))

        final_field[:, image == -1.0] = 0.0

        # Adjust the minimum and maximum values
        new_range = random.uniform(0.8, 1.0)
        image = image * new_range

        warped_image = self.composer.apply_to_image(image, final_field * scale)

        # Make images (3, H, W)
        image = np.array([image, image, image])
        warped_image = np.array([warped_image, warped_image, warped_image])

        return image, warped_image, final_field


# %% MARK: Dataset Visualizations


def _visualise_dataset(save=True):
    dataset = SyntheticDataset()
    fig, axes = plt.subplots(10, 4, figsize=(10, 24))
    for ax in axes.flatten():
        ax.axis("off")

    # images, motion = dataset._generate(0, 0)
    for i in range(10):
        image1, image2, motion = dataset._generate(0, 0, scale=(i + 1) / 5)
        diff = np.abs(image1 - image2)

        axes[i, 0].set_title(f"Scale: {(i + 1)/5:.2f}")
        axes[i, 0].imshow(image1, vmin=-1, vmax=1)
        axes[i, 1].imshow(image2, vmin=-1, vmax=1)
        axes[i, 2].imshow(flow_to_color(motion))
        axes[i, 3].imshow(diff)

    if save:
        os.makedirs(f"testing_visuals/dataset", exist_ok=True)
        plt.savefig(f"testing_visuals/dataset/dataset.png")


# %% MARK: Model


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        batch_dim = 0  # Satisfy linter
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convr1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convq1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )

        self.convz2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convr2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convq2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device)
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def bilinear_sampler(img, coords, mode="bilinear"):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), dim=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()
        self.args = Namespace()

        self.args.dropout = 0
        self.args.small = False
        self.args.mixed_precision = False
        self.args.alternate_corr = False

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.args.corr_levels = 4
        self.args.corr_radius = 4

        # feature network, context network, and update block
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn="instance", dropout=self.args.dropout
        )
        self.cnet = BasicEncoder(
            output_dim=hdim + cdim, norm_fn="batch", dropout=self.args.dropout
        )
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, (3, 3), padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(
        self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False
    ):
        """Estimate optical flow between pair of frames"""
        # image1: Tensor[B, 3, H, W]
        # image2: Tensor[B, 3, H, W]

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        return flow_predictions  # List[Tensor[B, 2, H, W]]


# %% MARK: Create Dataloader
training_dataset = SyntheticDataset()
validation_dataset = SyntheticDataset(validation=True)

training_dataloader = DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
)

print(f" === Created DataLoaders ({SyntheticDataset.VERSION}) ===\n\n")

# %% MARK: Create Model

model = RAFT().to(device)
if os.path.exists(SNAPSHOT_FILE):
    model.load_state_dict(torch.load(SNAPSHOT_FILE, weights_only=True))

RAFT_PARAMS = {
    "corr_levels": 4,
    "corr_radius": 4,
    "hidden_dim": 128,
    "context_dim": 128,
}

print(f" === Created Model ({RAFT.__name__}) ===")
print(model)
print("\n\n")

# %% MARK: Loss Function
LOSS_FUNCTION = "Sequence-EPE"
MAX_FLOW = 400


def loss_function(flow_preds: Tensor, flow_gt: Tensor, gamma=0.8, max_flow=MAX_FLOW):
    """Loss function defined over sequence of flow predictions"""

    n_predictions = len(flow_preds)
    flow_loss = torch.tensor(0.0).to(flow_gt.device)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


# %% MARK: Visualize Model


def _visualise_model(
    idx: int = 0, seed: int = 0, examples=1, didx=1, dseed=1, save=True
):
    return_figs: List[Tuple[str, Figure]] = []

    for i in range(examples):
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))

        for ax in axes.flatten():
            ax.axis("off")

        image1, image2, motion = validation_dataset._generate(
            idx + (didx * i), seed + (dseed * i)
        )

        batch_image1 = torch.from_numpy(image1).unsqueeze(0).to(device)
        batch_image2 = torch.from_numpy(image2).unsqueeze(0).to(device)
        batch_vectors = torch.from_numpy(motion).unsqueeze(0).to(device)

        print(image1.shape)
        print(image1.max())
        print(image1.min())

        model.eval()
        with torch.no_grad():
            prediction = model(batch_image1, batch_image2)

        loss, metrics = loss_function(prediction, batch_vectors)

        predicted_vectors = prediction[-1].squeeze(0).cpu().numpy()

        axes[0].set_title(f"Original Val idx: {idx}")
        axes[0].imshow(image1[0], vmin=-1, vmax=1)
        axes[1].set_title("Morphed")
        axes[1].imshow(image2[0], vmin=-1, vmax=1)
        axes[2].set_title(f"Vector Field seed: {seed}")
        axes[2].imshow(flow_to_color(motion))
        axes[3].set_title("Prediction")
        axes[3].imshow(flow_to_color(predicted_vectors))
        axes[4].set_title(f"Diff (EPE: {metrics['epe']:.4f})")
        axes[4].imshow(flow_to_color(predicted_vectors - motion))

        plt.tight_layout()

        return_figs.append((f"{idx + didx * i}_{seed + dseed * i}", fig))

    if save:
        os.makedirs(f"testing_visuals/model", exist_ok=True)
        plt.savefig(f"testing_visuals/model/model.png")

    plt.show()

    return return_figs


# %% MARK: Optimizer

OPTIMIZER_OPTIONS = {
    "type": "AdamW",
    "learning_rate": 0.0004,
    "weight_decay": 0.0001,
    "epsilon": 1e-08,
}

optimizer = optim.AdamW(
    model.parameters(),
    lr=OPTIMIZER_OPTIONS["learning_rate"],
    weight_decay=OPTIMIZER_OPTIONS["weight_decay"],
    eps=OPTIMIZER_OPTIONS["epsilon"],
)

SCHEDULER_OPTIONS = {
    "type": "ReduceLROnPlateau",
    "mode": "min",
    "factor": 0.1,
    "patience": 5,
}

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode=SCHEDULER_OPTIONS["mode"],
    factor=SCHEDULER_OPTIONS["factor"],
    patience=SCHEDULER_OPTIONS["patience"],
)

# %% MARK: Weights and Biases Setup
import wandb

wandb_config = {
    "gpu": GPU,
    "gpu_memory": GPU_MEMORY,
    "batch_size": BATCH_SIZE,
    "architecture": RAFT.__name__,
    "model_params": RAFT_PARAMS,
    "dataset": {
        "version": SyntheticDataset.VERSION,
        "training_tiles": NUM_TRAINING_TILES,
        "validation_tiles": NUM_VALIDATION_TILES,
    },
    "loss_function": LOSS_FUNCTION,
    "optimizer": OPTIMIZER_OPTIONS,
    "scheduler": SCHEDULER_OPTIONS,
}

print(" === Run Config ===\n")
print(wandb_config)
print("\n")

starting_batch = 0
run = None

if WANDB_ENABLED:
    run = wandb.init(
        project="motion-model",
        name=RUN_NAME,
        config=wandb_config,
        resume="auto",
    )

    from wandb.apis.public.runs import Run

    api = wandb.Api()
    existing_run: Run = api.run(run.path)
    history: List[Dict[str, int]] = existing_run.history(
        samples=1, pandas=False, keys=["snapshot_batch"]
    )
    if len(history) > 0:
        starting_batch: int = history[0]["snapshot_batch"]

    run.alert(
        title=f"Start Training",
        text=f"Starting training for {RUN_NAME}. Batch size: {BATCH_SIZE}, GPU Memory: {GPU_MEMORY}, GPU: {GPU}",
        level="INFO",
    )

    wandb.watch(model)
    print(run.name)
else:
    print("WANDB disabled")

# %% MARK: Training Loop

batch = starting_batch

training_start_time = time.time()
last_save_snapshot_time = time.time()

print("\n\n")
print("=== Starting Training ===")
print("- Batch Size: ", BATCH_SIZE)
print("- GPU Memory: ", GPU_MEMORY)
print("- GPU: ", GPU)
print("- Starting Batch: ", starting_batch)
print("- Starting Training: ", training_start_time)
print("- Last Save Snapshot: ", last_save_snapshot_time)
print("- Evaluation Frequency: ", EVALUATION_FREQUENCY)
print("- Snapshot Frequency: ", SNAPSHOT_FREQUENCY)
print("\n\n")

for idx, (batch_image1, batch_image2, batch_vectors) in enumerate(training_dataloader):
    batch += 1

    model.train()

    batch_image1, batch_image2, batch_vectors = (
        batch_image1.to(device),
        batch_image2.to(device),
        batch_vectors.to(device),
    )

    pred = model(batch_image1, batch_image2)
    loss, metrics = loss_function(pred, batch_vectors)
    loss.backward()

    total_grads = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    optimizer.zero_grad()

    log: Dict[str, Any] = defaultdict(dict)
    log["batch"] = batch
    log["training_metrics"] = {
        "loss": loss.item(),
        "epe": metrics["epe"],
        "1px": metrics["1px"],
        "3px": metrics["3px"],
        "5px": metrics["5px"],
    }
    log["gradient_norm"] = total_grads
    log["learning_rate"] = optimizer.param_groups[0]["lr"]

    log_examples = False

    model.eval()

    # Only evaluate every EVALUATION_FREQUENCY batches
    if batch % EVALUATION_FREQUENCY == 0:
        print(f"[{time.time() - training_start_time:.2f}s | {batch}] Evaluating...")
        samples = 0
        validation_losses = 0.0
        epes = 0.0
        epes_1px = 0.0
        epes_3px = 0.0
        epes_5px = 0.0

        with torch.no_grad():

            for batch_image1, batch_image2, batch_vectors in validation_dataloader:
                batch_image1 = batch_image1.to(device)
                batch_image2 = batch_image2.to(device)
                batch_vectors = batch_vectors.to(device)

                pred = model(batch_image1, batch_image2)
                loss, metrics = loss_function(pred, batch_vectors)

                samples += 1
                validation_losses += loss.item()
                epes += metrics["epe"]
                epes_1px += metrics["1px"]
                epes_3px += metrics["3px"]
                epes_5px += metrics["5px"]

        average_validation_loss = validation_losses / samples
        scheduler.step(average_validation_loss)

        average_epes = epes / samples
        average_epes_1px = epes_1px / samples
        average_epes_3px = epes_3px / samples
        average_epes_5px = epes_5px / samples

        log["validation_metrics"] = {
            "loss": average_validation_loss,
            "epe": average_epes,
            "1px": average_epes_1px,
            "3px": average_epes_3px,
            "5px": average_epes_5px,
        }

        if WANDB_ENABLED and run:
            log_examples = True

    # Only save every SNAPSHOT_FREQUENCY hours
    if time.time() - last_save_snapshot_time > SNAPSHOT_FREQUENCY * 60 * 60:
        print(
            f"[{time.time() - training_start_time:.2f}s | {batch}] Saving snapshot..."
        )
        last_save_snapshot_time = time.time()

        with torch.no_grad():
            torch.save(model.state_dict(), SNAPSHOT_FILE)
            if WANDB_ENABLED and run:
                model_artifact = wandb.Artifact(
                    name=f"motion_vector_model_{run.id}",
                    type="model",
                    description="Trained motion vector model",
                )
                model_artifact.add_file(SNAPSHOT_FILE)
                run.log_artifact(model_artifact)
                log["snapshot_batch"] = batch

                log_examples = True

    if log_examples:
        with torch.no_grad():
            samples = []
            for example_id, example in enumerate(EXAMPLE_TILES):
                example_path, example_images = example
                image1, image2 = example_images
                batch_image1 = image1.unsqueeze(0).to(device)
                batch_image2 = image2.unsqueeze(0).to(device)

                example_pred = model(batch_image1, batch_image2)
                loss, metrics = loss_function(example_pred, batch_vectors)

                fig, axes = plt.subplots(1, 3, figsize=(12, 5))

                for ax in axes.flatten():
                    ax.axis("off")

                predicted_vectors = example_pred[-1].squeeze(0).cpu().numpy()

                fig.suptitle(f"Batch {batch}")

                axes[0].set_title(f"Original Val idx: {idx}")
                axes[0].imshow(image1[0], vmin=-1, vmax=1)
                axes[1].set_title("Morphed")
                axes[1].imshow(image2[0], vmin=-1, vmax=1)
                axes[2].set_title("Prediction")
                axes[2].imshow(flow_to_color(predicted_vectors))

                plt.tight_layout()

                samples.append(wandb.Image(fig, caption=f"Example {example_path}"))
            log["images"]["examples"] = samples

            figs = _visualise_model(idx=0, examples=8, didx=20, dseed=1, save=False)
            figs = [wandb.Image(fig, caption=f"Example {id}") for id, fig in figs]
            log["images"]["validations"] = figs

    if WANDB_ENABLED and run:
        run.log(log)

    plt.close("all")  # Clean up plot if they are still open

print("Done!")

# %%
