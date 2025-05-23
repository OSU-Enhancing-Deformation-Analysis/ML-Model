# Material Deformation Prediction Model - Training Pipeline

## 1. Overview

Welcome! This document provides a comprehensive guide to the Python code training pipeline for our material deformation prediction model.

**The Core Idea:**
This project trains a specialized neural network to predict how that image would look if it underwent a specific type_of deformation (like stretching, twisting, or a more complex, localized change).

*   **Input:** The model takes two images:
    1.  The original, undeformed image.
    2.  The same image, but *after* a known deformation has been applied to it synthetically (in the code this is called the "morphed" or "warped" image). 
*   **Task:** The model learns to predict the "vector field" (displacement) that transformed the original image into the morphed image.
*   **Output:** A trained model that, when given a new pair of original and morphed images, can estimate the deformation field. This field can then be used in other applications, such as the "Model Output Preview" tool, to visualize or analyze material changes.

**Why this approach?**
By training on many examples of known deformations, the model learns the underlying patterns of how materials change. This allows it to:
1.  Estimate deformations even when we don't know them beforehand.
2.  Potentially identify subtle or complex deformation patterns that are hard to see by eye.

This document will guide you through:
*   Setting up your environment.
*   Preparing your input images.
*   Running the training process.
*   Understanding and configuring the various options.
*   Using the trained model with the "Model Output Preview" tool.
*   Making modifications and improvements to the pipeline.

## 2. Key Features of This Pipeline

*   **Image Tiling:** The code handles large input images by breaking them into smaller, manageable tiles for training.
*   **Synthetic Data Generation:** An infinite dataset is created by:
    *   Applying various mathematical "vector fields" (describing deformations) to input images.
    *   Using "shape masks" to apply deformations selectively to specific regions of an image.
    *   Randomly flipping, rotating, and adjusting image properties to make the model more robust.
*   **Efficient Data Handling:** Uses PyTorch's `IterableDataset` for memory-efficient data loading, especially useful for this large or virtually infinite dataset.
*   **Neural Network Architecture:** Includes a U-Net like convolutional neural network with self-attention mechanisms to capture both local and global features in the images.
*   **Training:** Uses the AdamW optimizer and a learning rate scheduler for effective training and loops through batches of images to train the model on the synthetic dataset.
*   **Experiment Tracking:** Integrated with Weights & Biases (WandB) for logging metrics, configurations, and visualizations (optional).
*   **Configuration Options:** Many aspects of the pipeline can be configured via command-line arguments or by editing constants directly in the script.

## 3. Prerequisites and Setup

Before you begin, you'll need to set up your Python environment.

1.  **Python:** This code is designed for Python 3.10 or newer because thats what I had installed.
2.  **Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    This is a general list of libraries that are used in the code.
    *   `torch` (PyTorch): The core machine learning framework.
    *   `torchvision`, `torchaudio`: PyTorch companion libraries.
    *   `numpy`: For numerical operations.
    *   `matplotlib`: For plotting and creating visualizations.
    *   `Pillow` (PIL): For image manipulation.
    *   `pyfastnoisesimd`: For fast Simplex noise generation (doesn't work on MacOS).
    *   `scipy`: For scientific computing tasks (like image filtering).
    *   `skimage` (scikit-image): For image processing algorithms.
    *   `flow_vis`: For visualizing vector fields as color images.
    *   `wandb`: For Weights & Biases integration (optional).
    *   `argparse`: For handling command-line arguments (usually built-in).

    Example installation for PyTorch (check [pytorch.org](https://pytorch.org) for specific commands for your system, especially if you have an NVIDIA GPU for CUDA support):
    ```bash
    # Example for CUDA 11.8 (check PyTorch website for current recommendations)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install numpy matplotlib Pillow pyfastnoisesimd scipy scikit-image flow_vis wandb
    ```
3.  **GPU (Highly Recommended):** Training neural networks is very computationally intensive. An NVIDIA GPU with CUDA support will drastically speed up training. The script will automatically try to use a GPU if one is available and correctly configured. If not, it will run on the CPU, which will be much slower. 
4.  **Input Data:**
    *   **Training Images:** Place your raw material images (e.g., `.tif` files) in a directory. The path to this directory is specified by the `IMAGES_DIR` variable in the script or the `--images_dir` command-line argument.
    *   **Example Images (for visualization during training):** Place pairs of smaller test images (e.g., `image_a.png`, `image_b.png`, `test_a.png`, `test_b.png`) in a separate directory. These are used to generate visual examples of the model's performance during training and must be the same size as the `TILE_SIZE` variable. The path is specified by `EXAMPLE_IMAGES_DIR` or `--example_images_dir`.

## 4. Understanding the Code (`m5-combo.py`)

The main logic is contained in a single Python script, `m5-combo.py`. The file is very long, but broken down into sections. Here is a detailed look at each section:

*   [**`MARK: Imports`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L23-L60):
    *   **What it does:** Loads all the necessary external Python libraries (like PyTorch, NumPy, etc.) that the script needs to function.
    *   **Quirks:** There is a patch to `numpy` that sets the `np.product` function to use `np.prod`. This is needed for the `pyfastnoisesimd` library which is very old and uses the deprecated `np.product` function.

*   [**`MARK: Constants`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L61-L96):
    *   **What it does:** Defines global settings and default values for the script. This holds a lot of the configuration options for training the model.
    *   **Key constants here:**
        *   `device`: Automatically detects if a GPU (CUDA) is available and sets the script to use it. Otherwise, defaults to CPU.
        *   `BATCH_SIZE`: Determines how many image pairs are processed by the model at once during training. This is often limited by GPU memory. The script attempts to calculate a reasonable default based on your GPU memory. This calculation is quite platform-specific, so it is best to adjust it using trial and error. A good batch size is one that fits in your GPU memory.
        *   `NUM_WORKERS`: How many parallel processes to use for loading data. `0` usually means data loading happens in the main process. A good value is 2 while training.
        *   `SNAPSHOT_FREQUENCY`: How often (in hours) to save a "snapshot" (a checkpoint) of the trained model.
        *   `EVALUATION_FREQUENCY`: How often to evaluate the model's performance on a separate validation dataset. This is a percentage of the training set (e.g., 0.05 means evaluate after every 5% of the training set). If the value is over 1, it will be interpreted as the number of batches (e.g., 5 means evaluate after every 5 batches).
        *   `WANDB_ENABLED`: A switch to turn on/off logging to Weights & Biases.
        *   `RUN_NAME`: A name for your training run, used for saving files and for WandB.

*   [**`MARK: Tile Loading`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L97-L126):
    *   **What it does:** Defines settings related to how your input images are processed, especially if they need to be "tiled" before being fed into the model.
    *   **Key constants here:**
        *   `IMAGES_DIR`: Path to your main training images. This can be a directory containing pre-cut tiles or a directory containing full images that need to be tiled by the script. It should be an absolute path or a path relative to where the script is run.
        *   `IMAGES_FILE_EXTENSION`: e.g., `.tif`, `.png`.
        *   `DIR_CONTAINS_TILES`: Set to `True` if your `IMAGES_DIR` already contains pre-cut tiles that are `TILE_SIZE`. Set to `False` if it contains large, full images that need to be tiled by the script.
        *   `EXAMPLE_IMAGES_DIR`: Path to your example images for visualization. These are used to generate visual examples of the model's performance during training and must be the same size as the `TILE_SIZE` variable. It should be an absolute path or a path relative to where the script is run.
        *   `TILE_SIZE`: The dimension (in pixels, e.g., 256x256) of the square tiles to be created or used. This is used to contruct the model's architecture so it shouldn't be changed during training.
        *   `MAX_TILES`: A limit on the total number of tiles to use from your dataset. By default, this is set to 100,000 tiles.
        *   `VALIDATION_SPLIT`: The percentage of tiles to set aside for validation (checking model performance on unseen data).
        *   `OVERLAP_SIZE`: When tiling full images, this is how much overlap (in pixels) there will be between adjacent tiles. Overlap helps ensure features near tile edges are seen in multiple contexts. A higher value will mean more tiles are created giving the model more training data.
        *   `INCLUDE_OUTSIDE`: If `True` when tiling, tiles can include black regions from outside the original image boundaries. If `False`, tiles are adjusted to stay within image boundaries.
    *   **Why:** Tiling is important for this synthetic dataset because all the "vector fields" and "shape masks" are calibrated for a specific tile size. While most Convolutional models can work on any image resolution without adjustment, this is not one of them due to the self-attention and the synthetic data it is trained on. Overlapping tiles prevent information loss at edges. Validation split ensures an unbiased measure of performance.

*   [**`MARK: Load CLI Options`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L127-L272):
    *   **What it does:** Sets up a system to allow you to change many of the above constants and settings from the command line when you run the script, without needing to edit the file itself.
    *   **Why:** This is convenient for running multiple experiments with different settings without constantly modifying the code. Any command-line option will override the default value set in the "Constants" or "Tile Loading" sections.

*   [**`MARK: Load Tiles`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L273-L509):
    *   **What it does:** This section actually performs the image loading and tiling.
        *   It scans the `IMAGES_DIR` for image files.
        *   If `DIR_CONTAINS_TILES` is `False`, it calls `create_tile_references` for each full image. This function doesn't load the full images into memory at once; instead, it calculates the coordinates of all possible tiles.
        *   The actual tile pixel data is loaded on-demand by `get_tile_from_reference` when needed during training.
        *   It splits the list of tiles into training and validation sets.
        *   It also loads the example images from `EXAMPLE_IMAGES_DIR`.
    *   **Decision Rationale (`create_tile_references` vs. `create_tiles`):**
        *   `create_tiles` Would load an entire image and cut all its tiles into memory. This is only feasible is using a small number of images and you can afford the memory cost. Honestly, I don't think it's worth it.
        *   `create_tile_references`: This function is more memory efficient for very large datasets. It just stores where tiles *are* in the original images. The actual image data for a tile is read from disk only when that specific tile is needed. This is the active method if `DIR_CONTAINS_TILES` is `False`.
    *   **How tiling works (`create_tile_references` and `get_tile_from_reference`):**
        *   The goal of tiling is to get a tile centered on as many pixels as possible and it does this by moving a grid over the image. While you can get a tile at every pixel, this is overkill if you have a variety of images.
        *   `SHIFT_SIZE = TILE_SIZE - OVERLAP_SIZE` determines how much the grid moves for each new tile.
        *   If `INCLUDE_OUTSIDE` is `True`, tiles can start partially off the image, padding with black.
        *   If `False`, tiles are adjusted to stay within the image boundaries.
        *   `get_tile_from_reference` takes an image index and (x, y) coordinates, opens the original image, crops the `TILE_SIZE x TILE_SIZE` region, and pads if necessary (e.g., if the tile is at the very edge of the image and smaller than `TILE_SIZE`).

*   [**`MARK: Vector Fields`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L510-L758):
    *   **What it does:** Defines different mathematical functions that describe deformations. These are the "known deformations" we apply to original images to create the morphed training pairs.
        *   Each function (e.g., `translation_field`, `rotation_field`, `perlin_field`) takes X and Y coordinate grids and returns dX and dY displacement values for each point.
        *   The `@vector_field()` decorator automatically registers these functions in a dictionary `VECTOR_FIELDS` and casts them to `np.float32`.
        *   `VectorField` class: A wrapper that allows a base field to be randomized (scaled, rotated, translated, amplitude changed).
        *   `VectorFieldComposer`: A class to manage and combine multiple `VectorField` instances. It can:
            *   Add different types of fields.
            *   Randomize their parameters.
            *   Sum their effects to create a complex, composite field.
            *   Apply the final combined field to an image using `map_coordinates` (this is the "warping" step, using bilinear interpolation `order=1` for smoothness).
    *   **Why:** We need a diverse set of deformations for the model to learn effectively. The composer allows us to create very complex and varied fields by combining simpler ones. Randomization ensures the model sees many variations.
    *   **Decision behind `pyfastnoisesimd` for Perlin/Simplex noise:** This library is much faster than other Python noise libraries, speeding up data generation. Simplex noise is generally considered to have fewer directional artifacts than classic Perlin noise. The only downside is that it doesn't work on MacOS.
    *   **Fields are (2, H, W) arrays:** This stores the dX and dY components as two separate channels in the same array `[ dX, dY ]`.

*   [**`MARK: Flow Visualizations`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L759-L941):
    *   **What it does:** Contains functions (`flow_to_color`, `flow_to_RG_color`) to convert the 2D vector fields (dX, dY) into color images that are easier for us to interpret. This uses the `flow_vis` library. 
    *   Also includes `_visualise_vector_field` and `_visualise_all_vector_fields` which were used during development to test and debug the vector field generation. They produce plots showing random samples of a field, histograms of displacements, and average displacement maps. These are saved to `testing_visuals/vector_fields/`. 

*   [**`MARK: Shape Masks`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L942-L1303):
    *   **What it does:** Defines functions to create various 2D shapes (square, circle, blob, gradient, etc.) as images where pixel values range from 0 to 1. These are called "masks."
        *   Each function (e.g., `create_square_shape`, `create_perlin_noise_shape`) generates a `TILE_SIZE x TILE_SIZE` array.
        *   These shapes are scaled, rotated, and translated randomly.
        *   All shapes are normalized to a 0-1 range.
    *   **Why:** These masks are used in the dataset generation to apply deformations (or different deformations) to only parts of an image, simulating localized effects or distinct regions. This makes the training data more complex and realistic.

*   [**`MARK: Shape Visualizations`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L1304-L1360):
    *   **What it does:** Similar to vector field visualizations, `_visualise_shape` and `_visualise_all_shapes` were used for debugging the shape generation functions. They save example images to `testing_visuals/shape_masks/`. You can run them to preview what the shapes look like.

*   [**`MARK: Dataset` (`SyntheticDataset` class)**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L1361-L1492):
    *   **What it does:** This is the heart of the data generation. It's an `IterableDataset`, meaning it generates data on-the-fly rather than storing everything in memory or in files.
    *   **The `_generate` method:**
        1.  **Load Base Image:** Gets an original image tile using `get_tile_from_reference`.
        2.  **Image Preprocessing:**
            *   Normalizes pixel values to 0.0-1.0.
            *   Applies random rotations (0, 90, 180, 270 degrees) and flips (horizontal/vertical).
            *   Randomly adjusts brightness/contrast by clipping and rescaling values.
        3.  **Generate Primary Vector Field:**
            *   Chooses 1 or 2 random vector field types (e.g., "rotation" + "perlin").
            *   Randomizes their parameters (center, scale, rotation, amplitude) using `composer.add_field()`.
            *   Combines them into `computed_field`.
        4.  **Optional Shape Masking (75% chance):**
            *   If a shape mask is used:
                *   A random shape function is chosen (e.g., `create_circle_shape`).
                *   The shape mask (0-1 values) is generated.
                *   **Inversion (50% chance):** Mask becomes `1 - mask`.
                *   **Blurring (50% chance):** `gaussian_filter` is applied to soften edges.
                *   **Mask Warping:** Another random vector field is generated and applied *to the shape mask itself*, distorting the mask.
                *   **Field Blending:**
                    *   `field_a = computed_field` (the primary field).
                    *   `field_b` is either `-computed_field` (inverted primary field, 50% chance) OR a completely new random field (50% chance).
                    *   `final_field = field_a * warped_mask + field_b * (1 - warped_mask)`. This means `field_a` is applied "outside" the mask (where mask is close to 1 after potential inversion) and `field_b` is applied "inside" (where mask is close to 0).
        5.  **No Shape Mask (25% chance):** `final_field = computed_field`.
        6.  **Zero out field where image is black:** If the original tile has pure black regions (e.g., from padding), the deformation field in those regions is set to zero.
        7.  **Apply Final Field:** The `final_field` is applied to the preprocessed original image to create the `warped_image` using `composer.apply_to_image()`.
        8.  **Return:** The original image, the warped image, and the `final_field`.
    *   **Why so much randomization:** This creates a very diverse training set from a smaller set of base images. It helps the model generalize better to unseen data and become robust to variations in input. Part of the reason for the ItterableDataset is to allow for very large datasets that are just randomly generated on the fly.

*   [**`MARK: Dataset Visualizations`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L1493-L1517):
    *   **What it does:** `_visualise_dataset` generates and plots a few examples from the `SyntheticDataset` to show the original image, the warped image, the true vector field, and the difference image. Saved to `testing_visuals/dataset/`.
    *   **Why:** Helps verify the entire data generation pipeline.

*   [**`MARK: Model` (`ComboMotionVectorConvolutionNetwork` class)**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L1518-L1696):
    *   **What it does:** Defines the architecture of the neural network. It's a U-Net like structure:
        *   **Encoder (Downsampling Path):** `conv1`, `pool1`, `conv2`, `pool2`, `conv_layers_down`.
            *   Consists of `ConvolutionBlock` modules (convolution, batch normalization, LeakyReLU activation) and `MaxPool2d` layers.
            *   Gradually reduces the spatial size (height, width) of the input while increasing the number of feature channels (depth). This helps capture increasingly abstract features.
            *   Skip connections: Features from these layers (`conv1`, `conv2`, and intermediate outputs from `conv_layers_down`) are saved to be concatenated later in the decoder.
        *   **Bottleneck:** `attention = SelfAttention(channels)`.
            *   A `SelfAttention` module is applied at the deepest part of the network. This helps the model weigh the importance of different regions of the feature map, capturing long-range dependencies.
        *   **Decoder (Upsampling Path):** `upconv1` through `upconv4`, `conv_up1` through `conv_up4`.
            *   Uses `ConvTranspose2d` (sometimes called "deconvolution") to increase spatial size.
            *   Concatenates features from the corresponding encoder layer (skip connections) with the upsampled features. This helps preserve high-resolution details.
            *   Applies `ConvolutionBlock`s to refine features.
        *   **Output Layer:** `output_conv`. A final convolution that reduces the channels to 2 (for dX and dY of the predicted vector field).
    *   **Why this architecture (U-Net with Attention):**
        *   **U-Nets** are very effective for image-to-image tasks (like predicting a field from an image pair) because the skip connections allow the network to combine low-level details with high-level semantic information.
        *   **Self-Attention** can improve performance by allowing the model to focus on relevant parts of the input across larger distances than standard convolutions can easily manage.
        *   **`ConvolutionBlock` with residual connection:** The `self.conv(x) + self.residual(x)` pattern helps with training deeper networks by making it easier for gradients to flow.

*   [**`MARK: Create Dataloader`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L1697-L1712):
    *   **What it does:** Wraps the `SyntheticDataset` in PyTorch `DataLoader` objects.
    *   `DataLoader` handles:
        *   Batching: Grouping individual data samples into batches (size `BATCH_SIZE`).
        *   Shuffling is implicitly handled by the `IterableDataset`'s randomness.
        *   Parallel data loading (if `NUM_WORKERS > 0`).
        *   `pin_memory=True`: Can speed up data transfer to GPU.

*   [**`MARK: Create Model`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L1713-L1722):
    *   **What it does:** Creates an instance of the `ComboMotionVectorConvolutionNetwork` and moves it to the `device` (GPU or CPU).
    *   It also checks if a snapshot file (e.g., `RUN_NAME.pth`) exists and loads its weights if present. This allows resuming training from a previous checkpoint.
    
*   [**`MARK: Loss Function` (`loss_function`)**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L1723-L1745):
    *   **What it does:** Defines how to measure the "error" or "difference" between the model's predicted vector field (`flow_preds`) and the true vector field (`flow_gt`).
    *   The primary loss is based on the absolute difference: `(flow_preds - flow_gt).abs().mean()`.
    *   It also calculates **End-Point Error (EPE)**: `torch.sum((flow_preds - flow_gt)**2, dim=1).sqrt()`. This is the Euclidean distance between the predicted and true displacement for each pixel.
    *   It also computes metrics like "1px", "3px", "5px" error: the percentage of pixels where EPE is less than 1, 3, or 5 pixels, respectively.
    *   **Why:** The loss function guides the training process. The model tries to adjust its internal parameters (weights) to minimize this loss value. EPE and N-pixel errors are standard metrics for evaluating optical flow / motion vector prediction.
    *   **How to change loss function:** You would modify the `loss_function` Python function. Ensure it returns a single scalar value for the loss (used for backpropagation) and a dictionary of metrics for logging. For example, if you wanted to use Mean Squared Error (MSE) as the primary loss:
        ```python
        def loss_function(flow_preds: Tensor, flow_gt: Tensor):
            # Old:
            # i_loss = (flow_preds - flow_gt).abs()
            # flow_loss = (i_loss).mean()

            # New for MSE:
            flow_loss = F.mse_loss(flow_preds, flow_gt) # Or ((flow_preds - flow_gt)**2).mean()
            epe = torch.sum((flow_preds - flow_gt) ** 2, dim=1).sqrt()

            epe = epe.view(-1)
            metrics = {
                "epe": epe.mean().item(),
                "1px": (epe < 1).float().mean().item(),
                "3px": (epe < 3).float().mean().item(),
                "5px": (epe < 5).float().mean().item(),
                "mse_loss": flow_loss.item() # if you want to log the main loss value as well
            }
            return flow_loss, metrics
        ```

*   [**`MARK: Visualize Model` (`_visualise_model`)**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L1746-L1798):
    *   **What it does:** Takes a sample from the validation dataset, runs it through the current model, and plots the original image, morphed image, true vector field, predicted vector field, and the difference between true and predicted fields.
    *   Used for generating example images for Weights & Biases logging during training.
    *   **Why:** Provides a qualitative assessment of the model's performance on unseen data.

*   [**`MARK: Optimizer`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L1799-L1821):
    *   **What it does:** Sets up the optimization algorithm.
        *   **Optimizer (`AdamW`):** An algorithm that updates the model's weights based on the calculated loss. AdamW is a popular and effective optimizer, an improvement over standard Adam. `lr` is the learning rate. This is where the backpropagation takes place.
        *   **Scheduler (`ReduceLROnPlateau`):** Adjusts the learning rate during training. If the validation loss stops improving for a certain number of evaluations (`patience`), the learning rate is reduced by a `factor`.
    *   **Why:** The optimizer is what actually "learns." The scheduler helps fine-tune the learning process, often leading to better results by reducing the learning rate as training converges.

*   [**`MARK: Weights and Biases Setup`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L1822-L1871):
    *   **What it does:** Initializes Weights & Biases (`wandb`) if `WANDB_ENABLED` is `True`.
        *   It logs the configuration (batch size, model architecture, etc.).
        *   It can resume logging from a previous run if using the same `RUN_NAME`.
        *   `wandb.watch(model)` tracks gradients and parameters of the model.
    *   **Why:** WandB is a powerful tool for tracking experiments, comparing different runs, and visualizing results, especially when you're trying out many different settings.

*   [**`MARK: Training Loop`**](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/m5-combo.py#L1872-L2028):
    *   **What it does:** This is the main loop where training happens.
        1.  Iterates through the `training_dataloader`, getting batches of `(batch_images, batch_vectors)`.
        2.  **Training Step:**
            *   `model.train()`: Sets the model to training mode (enables things like dropout, updates batch norm stats).
            *   `optimizer.zero_grad()`: Clears old gradients for the next iteration.
            *   Moves data to `device`.
            *   `pred = model(batch_images)`: Forward pass – gets the model's prediction.
            *   `loss, metrics = loss_function(pred, batch_vectors)`: Calculates loss.
            *   `loss.backward()`: Backward pass – calculates gradients (how much each neuron contributed to the loss).
            *   `torch.nn.utils.clip_grad_norm_`: Clips gradients to prevent them from becoming too large (exploding gradients), which can destabilize training.
            *   `optimizer.step()`: Updates model weights using the optimizer.
        3.  **Logging:** Records training loss, EPE, etc.
        4.  **Evaluation Step (periodically, based on `EVALUATION_FREQUENCY`):**
            *   `model.eval()`: Sets model to evaluation mode (disables dropout, uses fixed batch norm stats).
            *   `with torch.no_grad()`: Disables gradient calculations (saves memory and computation).
            *   Iterates through `validation_dataloader`, calculates loss and metrics on validation data.
            *   Averages validation metrics.
            *   `scheduler.step(average_validation_loss)`: Updates learning rate based on validation loss.
            *   Logs validation metrics to WandB.
        5.  **Snapshotting (periodically, based on `SNAPSHOT_FREQUENCY`):**
            *   `torch.save(model.state_dict(), SNAPSHOT_FILE)`: Saves the model's current weights to a `.pth` file.
            *   If WandB is enabled, logs this file as an "artifact."
        6.  **Example Image Logging (if evaluation or snapshot happened and WandB is on):**
            *   Runs the model on the `EXAMPLE_TILES` (from `EXAMPLE_IMAGES_DIR`).
            *   Runs `_visualise_model` to generate visualizations on validation data.
            *   Logs these images to WandB.
        7.  `plt.close("all")`: Closes any open Matplotlib figures to save memory.
    *   **Why:** This loop implements the standard "train-evaluate-save" cycle of machine learning.

## 5. How to Run the Training

You can run the training script from your terminal.

**Basic Command:**
```bash
python m5-combo.py
```
This will run with all the default settings defined in the script.

**Using Command-Line Arguments:**
You can override default settings using command-line arguments. To see all available options:
```bash
python m5-combo.py --help
```
This will display a list like:
```
usage: m5-combo.py [-h] [--run_name RUN_NAME] [--batch_size_multiplier BATCH_SIZE_MULTIPLIER] ...

options:
  -h, --help            show this help message and exit
  --run_name RUN_NAME   Name of the run used for Weights and Biases and file names
  --batch_size_multiplier BATCH_SIZE_MULTIPLIER
                        Multiply the batch size by this number
  --evaluation_frequency EVALUATION_FREQUENCY
                        How often to evaluate the model as a percentage of the training set. Values over 1 are interpreted as # of batches
  ... (and many more)
```

**Example: Custom Run Name and Batch Size Multiplier**

> There is a shell script called [`start_job.sh`](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/start_job.sh) that can be used to submit a job to a SLURM cluster. Take a look at the script to see how the command is constructed.
> 

```bash
python m5-combo.py --run_name "MyFirstExperiment" --batch_size_multiplier 0.5
```
This will:
*   Name the run "MyFirstExperiment".
*   Reduce the automatically calculated batch size by half.

**Key Configurable Options (via Command Line or In-Script Constants):**

| Option (CLI)                   | Constant (In-Script)        | Description                                                                                                | Why Change It?                                                                                                                               |
| :----------------------------- | :-------------------------- | :--------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| `--run_name`                   | `RUN_NAME`                  | Identifier for this training run. Used for filenames and WandB.                                            | To organize different experiments.                                                                                                           |
| `--batch_size_multiplier`      | (Modifies `BATCH_SIZE`)     | Multiplies the auto-calculated `BATCH_SIZE`. Use <1 if you run out of GPU memory, >1 if you have spare.       | GPU memory constraints. Larger batches can sometimes stabilize training but use more memory.                                                 |
| `--evaluation_frequency`       | `EVALUATION_FREQUENCY`      | How often to evaluate on validation set (as % of training set, or #batches if >1).                         | More frequent evaluation gives finer-grained insight but slows training. Less frequent is faster but coarser.                                |
| `--snapshot_frequency`         | `SNAPSHOT_FREQUENCY`        | How often (in hours) to save model checkpoints.                                                            | Balance between saving progress frequently and disk space / WandB artifact clutter.                                                          |
| `--validation_split`           | `VALIDATION_SPLIT`          | Percentage of data for validation (e.g., 0.05 for 5%).                                                     | Ensure enough validation data for reliable performance measure, but not too much to shrink training data.                                    |
| `--max_tiles`                  | `MAX_TILES`                 | Maximum number of tiles to use for training/validation.                                                    | To limit dataset size for quick tests or if you have too many tiles.                                                                       |
| `--images_dir`                 | `IMAGES_DIR`                | Path to your training images.                                                                              | To point to your specific dataset location.                                                                                                  |
| `--example_images_dir`         | `EXAMPLE_IMAGES_DIR`        | Path to your example images for logging.                                                                   | To use specific images for consistent visualization across runs.                                                                           |
| `--images_file_extension`      | `IMAGES_FILE_EXTENSION`     | File type of training images (e.g., ".tif").                                                               | If your images are not `.tif`.                                                                                                                |
| `--tile_size`                  | `TILE_SIZE`                 | Dimension of square tiles (pixels).                                                                        | Model architecture might be sensitive to this. Larger tiles see more context but use more memory.                                          |
| `--overlap_size`               | `OVERLAP_SIZE`              | Overlap between tiles when cutting full images.                                                            | Larger overlap means more tiles and better handling of edge features, but more redundant data.                                              |
| `--num_workers`                | `NUM_WORKERS`               | Number of parallel processes for data loading.                                                             | `>0` can speed up data loading if it's a bottleneck, but uses more CPU/RAM. Start with `0` or `1`, increase if needed.                     |
| `--wandb_enabled` / `--wandb_disabled` | `WANDB_ENABLED`             | Enable/disable Weights & Biases logging.                                                                   | Disable if you don't want to use WandB or are running quick local tests.                                                                    |
| `--dir_contains_tiles` / `--dir_contains_full_images` | `DIR_CONTAINS_TILES`        | Tells script if `IMAGES_DIR` has pre-cut tiles or full images.                                         | Set according to your data format.                                                                                                           |
| `--include_outside` / `--exclude_outside` | `INCLUDE_OUTSIDE`           | Whether tiles can include areas outside the original image (padded with black).                            | `include_outside` can generate more tiles, especially from edges/corners.                                                                    |

**Changing In-Script Constants Directly:**
For settings not available as command-line arguments, or if you prefer, you can directly edit their values in the `m5-combo.py` script, typically in the "MARK: Constants" or "MARK: Tile Loading" sections.

## 6. Training with SLURM (for High-Performance Clusters)

If you have access to a High-Performance Computing (HPC) cluster that uses SLURM as a job scheduler, you can run your training jobs there. This is useful for long training runs or when you need significant GPU resources.

**Example SLURM Job Script ([`start_job.sh`](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/start_job.sh)):**

> There is a shell script called [`start_job.sh`](https://github.com/OSU-Enhancing-Deformation-Analysis/ML-Model/blob/refs/heads/main/start_job.sh) that can be used to submit a job to a SLURM cluster. Take a look at the script to see how the command is constructed.
> 

**How to use the SLURM script:**
1.  **Customize:**
    *   Adjust `#SBATCH` directives (resources like memory, time, GPU type) based on your needs and cluster limits.
    *   Update module loading commands specific to your HPC.
    *   Set the correct path to your project directory (`cd ...`).
    *   Modify the `python m5-combo.py ...` command with your desired parameters.
2.  **Submit:**
    ```bash
    sbatch train_model.sbatch
    ```

**Important considerations for SLURM:**
*   **Paths:** Use absolute paths for directories (`IMAGES_DIR`, etc.) if running from SLURM, as the working directory might be different.
*   **Environment:** Ensure the Python environment (modules, virtual environment) is correctly set up within the SLURM script so it can find PyTorch and all other dependencies.
*   **WandB with SLURM:** WandB should generally work fine from SLURM jobs as long as the compute node has internet access. You will need to sign into WandB using the python before running the job. If not, WandB can be configured for offline mode (`wandb offline`).

## 7. Using Trained Models in ["Model Output Preview"](https://github.com/OSU-Enhancing-Deformation-Analysis/Model-Output-Preview)

Once your `m5-combo.py` script has successfully trained a model, it will save a file named `YOUR_RUN_NAME.pth` (e.g., `b5-unknown-combo-test.pth`). This `.pth` file contains the learned "weights" or parameters of your neural network.

The ["Model Output Preview"](https://github.com/OSU-Enhancing-Deformation-Analysis/Model-Output-Preview) tool is a separate project designed to take this trained `.pth` file and use it to:
1.  Load new images (that the model hasn't seen before).
2.  Predict the deformation field for these new images.
3.  Visualize the original image, the predicted deformation, and the strain of the material calculated from the predicted deformation.

**General Steps to Use a Trained Model in the Preview Tool:**

1.  **Locate your trained model file:** This will be the `.pth` file created by the training script (e.g., `MyExperiment.pth`).
2.  **Transfer the model file (if necessary):** If the Preview tool is on a different machine, copy the `.pth` file to a location accessible by the Preview tool.
3.  **Configure the Preview Tool:**
    *   The Preview tool as the required classes included to run this model architecture (m5-combo). All you need to do is place the new `MyExperiment.pth` file into the models folder. Rename it to match `m4-combo.pt` so that it can be recognized by the Preview tool.
    *   The Preview tool expects the `TILE_SIZE` to be 256x256 pixels. If you trained the model on a different size, you'll need to adjust the `TILE_SIZE` in the Preview tool's `machine_learning_model.py` file.

> There are further instructions in the ["Model Output Preview"](https://github.com/OSU-Enhancing-Deformation-Analysis/Model-Output-Preview) repository on how to add a new model architecture to the Preview tool.
> 

## 8. Making Modifications and Improvements

This pipeline is designed to be flexible. Here’s how you can modify different parts:

**A. How to Tile Input Images (Configuration)**
   *   **Already Tiled?**
        *   Set `DIR_CONTAINS_TILES = True` (in script) or use `--dir_contains_tiles` (CLI).
        *   Ensure `IMAGES_DIR` points to the directory of these tiles.
        *   The script will assume these tiles are already the correct `TILE_SIZE`.
   *   **Full Images to be Tiled by Script?**
        *   Set `DIR_CONTAINS_TILES = False` (in script) or use `--dir_contains_full_images` (CLI).
        *   `IMAGES_DIR` points to the directory of full images.
        *   Configure `TILE_SIZE`: The desired size of output tiles (e.g., 256 for 256x256 pixels).
        *   Configure `OVERLAP_SIZE`: How much overlap between adjacent tiles (e.g., 64 pixels). A larger overlap means more tiles and potentially better learning at edges, but also more processing. `SHIFT_SIZE` (derived as `TILE_SIZE - OVERLAP_SIZE`) controls the step.
        *   Configure `INCLUDE_OUTSIDE`:
            *   `True` (`--include_outside`): Tiles can extend beyond the image boundary, padded with black. This generates more tiles, especially from the edges.
            *   `False` (`--exclude_outside`): Tiles are adjusted to stay within image boundaries. Tiles at the very edge are shifted inwards to ensure full `TILE_SIZE` content.

**B. How to Change Loss Functions**
1.  **Locate the `loss_function` function** in `m5-combo.py` (around "MARK: Loss Function").
2.  **Understand its Signature:**
    ```python
    def loss_function(flow_preds: Tensor, flow_gt: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        # flow_preds: Model's predicted vector field (BatchSize, 2, H, W)
        # flow_gt: Ground truth vector field (BatchSize, 2, H, W)
        # Returns:
        #   - A single scalar Tensor representing the loss for backpropagation.
        #   - A dictionary of metrics (float values) for logging.
    ```
3.  **Implement Your New Loss:**
    *   You can use PyTorch's built-in loss functions from `torch.nn.functional` (often aliased as `F`) or `torch.nn`.
    *   Example: To switch the main loss to Mean Squared Error (MSE):
        ```python
        import torch.nn.functional as F # Make sure F is available

        def loss_function(flow_preds: Tensor, flow_gt: Tensor):
            # Calculate your primary loss for backpropagation
            # Example: Mean Squared Error
            main_loss = F.mse_loss(flow_preds, flow_gt)

            # Keep calculating other metrics for logging if desired
            epe = torch.sum((flow_preds - flow_gt) ** 2, dim=1).sqrt()
            epe_scalar = epe.mean().item() # .item() converts single-element tensor to Python float

            metrics = {
                'epe': epe_scalar,
                '1px': (epe < 1).float().mean().item(),
                '3px': (epe < 3).float().mean().item(),
                '5px': (epe < 5).float().mean().item(),
                'mse_loss': main_loss.item() # Log your main loss too
            }

            return main_loss, metrics
        ```
4.  **Considerations:**
    *   The choice of loss function can significantly impact training performance and the characteristics of the learned model.
    *   Ensure your new loss function is differentiable (most standard ones are).

**C. Adding New Vector Fields**
1.  **Go to the `MARK: Vector Fields` section.**
2.  **Define a new Python function** that takes two NumPy arrays, `X` and `Y` (representing normalized coordinate grids from -1 to 1), and returns two NumPy arrays, `dX` and `dY` (the displacement components).
    ```python
    # Example: A new hypothetical 'pinch_field'
    def my_new_pinch_field(X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        # X and Y are grids of coordinates, e.g., from np.linspace(-1, 1, TILE_SIZE)
        # Implement your logic to calculate dX and dY based on X and Y
        strength = 0.5
        dX = -strength * X * (X**2 + Y**2) # Example: pushes towards Y-axis, stronger near origin
        dY = -strength * Y * (X**2 + Y**2) # Example: pushes towards X-axis, stronger near origin
        return dX, dY
    ```
3.  **Decorate it and Add to `VECTOR_FIELDS`:** The script uses a decorator `@vector_field()` to automatically register new fields.
    ```python
    @vector_field() # This decorator handles registration
    def my_new_pinch_field(X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        # ... (your implementation from above) ...
        return dX.astype(np.float32, copy=False), dY.astype(np.float32, copy=False) # Ensure float32
    ```
    The `vector_field` decorator adds your function to the `VECTOR_FIELDS` dictionary, making it available for random selection in the `SyntheticDataset`.
4.  **Test (Optional but Recommended):** You can call the `_visualise_vector_field` function to visualize your new field and ensure it behaves as expected. View the visualization to see how your field behaves when averaged on a large scale to find any biases or artifacts.

**D. Adding New Shape Masks**
1.  **Go to the `MARK: Shape Masks` section.**
2.  **Define a new Python function** that takes an integer `size` (e.g., `TILE_SIZE`) and returns a NumPy array of shape `(size, size)` with float values between 0.0 and 1.0.
    ```python
    # Example: A new 'cross_shape'
    def create_cross_shape(size: int) -> npt.NDArray[np.float32]:
        shape_array = np.zeros((size, size), dtype=np.float32)
        thickness = int(size * 0.1)  # 10% thickness
        center = size // 2

        # Horizontal bar
        y_start_h = center - thickness // 2
        y_end_h = y_start_h + thickness
        shape_array[y_start_h:y_end_h, :] = 1.0

        # Vertical bar
        x_start_v = center - thickness // 2
        x_end_v = x_start_v + thickness
        shape_array[:, x_start_v:x_end_v] = 1.0

        # Random rotation and translation (optional, can be copied from other shape functions)
        # For simplicity, this example omits them, but they are good for diversity.
        # If you add rotation/translation, ensure values remain clipped/normalized to 0-1.

        return shape_array.astype(np.float32, copy=False) # Must be float32, 0-1
    ```
3.  **Add to `shape_functions` list:**
    ```python
    shape_functions: List[Callable[[int], npt.NDArray[np.float32]]] = [
        create_square_shape,
        create_circle_shape,
        # ... other existing shapes ...
        create_voronoi_pattern,
        create_cross_shape  # Add your new function here
    ]
    ```
4.  **Test (Optional but Recommended):** Call `_visualise_shape` to check your new shape.

**E. Modifying the Model Architecture (`ComboMotionVectorConvolutionNetwork`)**
   This is the most advanced modification and requires some understanding of neural network design.
1.  **Locate the `ComboMotionVectorConvolutionNetwork` class** definition in "MARK: Model".
2.  **Identify Components:**
    *   `ConvolutionBlock`: Basic building block. You could change kernel sizes, add/remove layers within it.
    *   `SelfAttention`: Attention mechanism. You could try different attention types or remove it.
    *   Encoder layers (e.g., `self.conv1`, `self.conv_layers_down`).
    *   Decoder layers (e.g., `self.upconv1`, `self.conv_up1`).
3.  **Making Changes:**
    *   **Channel Sizes:** Changing the `base_channels` or multipliers (e.g., `channels * 2`) will change the model's capacity (and memory usage). Ensure channel sizes match up between concatenating layers.
    *   **Number of Layers:** Adding/removing `ConvolutionBlock`s or entire down/up stages will change depth.
    *   **New Layer Types:** You could introduce other PyTorch layers (`nn.Dropout`, different activation functions, etc.).
4.  **CRITICAL Considerations:**
    *   **Input/Output Shape:** The model must still accept input of shape `(Batch, 2, TILE_SIZE, TILE_SIZE)` (for 2 input images) and output `(Batch, 2, TILE_SIZE, TILE_SIZE)` (for dX, dY).
    *   **Skip Connections:** If you change encoder/decoder structure, ensure the skip connections (`torch.cat([...])`) correctly match feature maps of compatible sizes. Mismatches here are a common source of errors.
    *   **Parameter Count:** Drastic changes can significantly alter the number of trainable parameters, affecting training time and generalization.
    *   **Saved Weights Incompatibility:** If you change the model architecture, previously saved `.pth` files for the OLD architecture will **NO LONGER BE COMPATIBLE**. You will need to retrain the model from scratch or write custom code to try and partially load weights (which is complex).
    *   **Compatibility with Preview Tool:** The "Model Output Preview" tool must also be updated with the *exact same* new model class definition to be able to load models trained with the new architecture.

## 9. Understanding "Why it Works That Way" (Decision Rationale)

Throughout the "Understanding the Code" section and "Making Modifications," I've tried to include "Why" and "Decision Rationale" points. Here's a summary of some key design choices:

*   **Synthetic Data Generation:** Real-world paired data (original image + exactly known deformation field + warped image) is very hard to obtain. Synthetic generation gives us full control and virtually infinite data. The complexity (multiple fields, shape masks, warping masks) aims to create diverse and challenging examples that push the model to learn robust features.
*   **IterableDataset:** For memory efficiency. Generating data on-the-fly avoids storing a massive dataset in RAM.
*   **U-Net Architecture:** Standard and effective for image-to-image tasks where spatial information and multi-scale features are important. Skip connections are crucial for combining high-level semantics with low-level details.
*   **Self-Attention:** To help the model capture long-range dependencies in the image features, which can be important for understanding global deformation patterns.
*   **Randomization (in Data Generation):** Extensive randomization (field types, parameters, shapes, image augmentations) is key to "data augmentation." It prevents the model from "memorizing" the training data and helps it generalize better to new, unseen images.
*   **Vector Field Normalization/Tuning:** Care was taken so that, on average, random fields don't just push everything in one direction or cause a net scaling. This is important for learning unbiased motion.
*   **Order=1 Interpolation for Warping:** When `map_coordinates` applies a vector field to an image, `order=1` (bilinear interpolation) is used. This provides a smooth transformation, avoiding the blocky artifacts of `order=0` (nearest neighbor).

> If order=0 is used then displacements with magnitudes less than 1 pixel will have no effect on the image and create large artifacts.where no part of the image moved while the model is being told that it did move.
> 

*   **AdamW Optimizer and ReduceLROnPlateau Scheduler:** These are generally robust and effective choices for training deep neural networks, often leading to good convergence without excessive manual tuning of learning rates.

This pipeline represents a sophisticated approach to a complex problem. The combination of carefully designed synthetic data and a powerful neural network architecture allows it to learn intricate deformation patterns.