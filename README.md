# CLIM-DL

## Installation

This repo uses python 3.12+

Steps to install necessary components:

- General: 
    Install required packages: `pip install -r requirements.txt`

- On the fly:
    Install required packages: `pip install -r requirements_on_the_fly.txt`

## Training model

This script trains various segmentation models for photoluminescence (PL) grain analysis using synthetic data generation.

## Overview

The training script supports multiple model architectures for segmentation tasks:
- Default Segmentation Model
- Segmentation Model V2
- UNet Segmentation Model
- Upscale Segmentation Model
- UNet Upscale Segmentation Model

## Usage

```bash
python train_script.py <model_name> [options]
```

### Required Arguments

- `name` - Name of the model to be trained. This will be used in the filename when saving the model.

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | enum | `segmentation` | Model architecture to train |
| `--batches` | int | 50 | Number of batches in each epoch step |
| `--epochs` | int | 20 | Number of epochs to run training for |
| `--existing-model` | str | None | Path to existing model to continue training (optional) |
| `--output-dir`, `-o` | str | `./saved_models/` | Directory path where the model will be saved |
| `--noise-data` | str | None | Path to additional noise data (TIFF format) for training |

### Model Types

The `--model` argument accepts the following values:

- `segmentation` - Default segmentation model
- `segmentation_v2` - Default segmentation with extra CNN layers
- `unet_segmentation` - UNet-based segmentation model
- `upscale_segmentation` - Segmentation model with upscaling
- `unetupscale_segmentation` - Combined UNet and upscaling model

## Examples

### Basic Training
```bash
python train_script.py my_model --epochs 30 --batches 100
```

### Training with UNet Architecture
```bash
python train_script.py unet_model --model unet_segmentation --epochs 50
```

### Continue Training from Existing Model
```bash
python train_script.py continued_model --existing-model ./saved_models/previous_model.model --epochs 10
```

### Training with Custom Noise Data
```bash
python train_script.py noise_trained_model --noise-data ./data/noise_sample.tif --output-dir ./my_models/
```

### Training Upscaling Model
```bash
python train_script.py upscale_model --model upscale_segmentation --batches 75 --epochs 25
```

## Output Files

The script generates the following output files in the specified output directory:

- `<model_name>.model` - The trained model file
- `<model_name>.epoch_l` - Epoch-wise loss values
- `<model_name>.all_l` - All training step loss values

## Configuration

Training data generation parameters, model architecture details, and loss function settings can be configured directly in the script. The script automatically handles synthetic data generation and model configuration based on the selected model type.