#!/bin/bash
# JOB HEADERS HERE
#SBATCH --job-name=m5-combo
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=100:00:00

# Define variables
RUN_NAME="m5-combo"
IMAGES_DIR="~/hpc-share/full_images/gRaw/"
EXAMPLES_DIR="~/hpc-share/example_images/"
PYTHON_ENV=~/hpc-share/capstone_model_training/bin/activate

# Updates the batch file before running
BATCH_FILE="batch-m5-combo.py"
GITHUB_URL="https://raw.githubusercontent.com/OSU-Enhancing-Deformation-Analysis/ML-Model/refs/heads/main/$BATCH_FILE"

wget -O "$BATCH_FILE" "$GITHUB_URL"

source "$PYTHON_ENV"

if [[ -e ./images_dir || -L ./images_dir ]]; then
    echo "Notice: './images_dir' already exists. Skipping symlink creation."
else  
    ln -s "$IMAGES_DIR" ./images_dir
fi
if [[ -e ./example_images_dir || -L ./example_images_dir ]]; then
    echo "Notice: './example_images_dir' already exists. Skipping symlink creation."
else  
    ln -s "$EXAMPLES_DIR" ./example_images_dir
fi

echo "Setup complete!"

python "$BATCH_FILE" --evaluation_frequency=0.5 --snapshot_frequency=1 --run_name="$RUN_NAME" \
--images_dir="./images_dir" --example_images_dir="./example_images_dir" --wandb_enabled=True