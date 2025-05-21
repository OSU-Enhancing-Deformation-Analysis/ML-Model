#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=m5-combo
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=100:00:00

# Define variables
RUN_NAME="m5-combo"
IMAGES_DIR="/nfs/stak/users/schmitia/hpc-share/full_images/gRaw/"
EXAMPLES_DIR="/nfs/stak/users/schmitia/hpc-share/example_images/"
PYTHON_ENV=/nfs/stak/users/schmitia/hpc-share/capstone_model_training/bin/activate

# Updates the batch file before running
BATCH_FILE="batch-m5-combo.py"
GITHUB_URL="https://raw.githubusercontent.com/OSU-Enhancing-Deformation-Analysis/ML-Model/refs/heads/main/$BATCH_FILE"

wget -O "$BATCH_FILE" "$GITHUB_URL"

source "$PYTHON_ENV"

echo "Setup complete!"

python "$BATCH_FILE" --evaluation_frequency=0.5 --snapshot_frequency=1 --run_name="$RUN_NAME" \
--images_dir="$IMAGES_DIR" --example_images_dir="$EXAMPLES_DIR" --dir_contains_tiles=False