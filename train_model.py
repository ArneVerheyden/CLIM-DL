import argparse

import torch
from torch.utils.data.dataset import Dataset

from enum import Enum

from models.psf import GuassionPSF
from models.segmentation_unet_model import PLSegmentationUnetModel
from models.segmentation_upscaling import PLSegmentationScalingModel
from models import PLSegmentationUnetScalingModel
from simulation.grain_PL_simulation import TrainingDataSimulationOptions
from torch_utils.dataset import GeneratedPLOutlineDataset
from torch_utils.transform import BackgroundRemovalNormalize, SkipFrames
from training import ModelTrainer
from models.segmentation_model import PLSegmentationModel

from torchvision import transforms

import os

class Model(Enum):
    DefaultSegmentation = 'segmentation'
    UnetSegmentation = 'unet_segmentation'
    UpscaleSegmentation = 'upscale_segmentation'
    UnetUpscaleSegmentation = 'unetupscale_segmentation'


class SegmentationTrainer(ModelTrainer):
    def step(self, input, labels):
        input = input.unsqueeze(2)

        predictions = self.model(input)
        loss = self.loss_function(predictions.squeeze(1), labels)

        return loss
    
def train_segmentation(args, model_class: PLSegmentationModel | PLSegmentationUnetModel):
    model = None
    if args.existing_model:
        model = model_class.load(args.existing_model)
    else:
        model = model_class(
            input_dim=1, 
            hidden_dim=10, 
            kernel_size=3, 
            num_layers=1,
            # n_start_unet_channels=2,
        )  

    dataset = get_training_data(args.batches)
    loss_function = get_loss_function(dataset)

    trainer = SegmentationTrainer(model, dataset, dataset, 0.005, loss_function)
    trainer.train(args.epochs)

    save_path = os.path.join(args.output_dir, f'{args.name}.model')
    trainer.save(save_path, overwrite=True)

def train_scaling_segmentation_model(args, model_class: PLSegmentationScalingModel):
    model = None
    if args.existing_model:
        model = model_class.load(args.existing_model)
    else:
        model = model_class(
            input_dim=1, 
            hidden_dim=10, 
            kernel_size=3, 
            num_layers=1,
            scaling=2,
            n_start_unet_channels=8,
        )

    dataset = get_training_data(args.batches, label_scaling=2)
    loss_function = get_loss_function(dataset)

    trainer = SegmentationTrainer(model, dataset, dataset, 0.005, loss_function)
    trainer.train(args.epochs)

    save_path = os.path.join(args.output_dir, f'{args.name}.model')
    trainer.save(save_path, overwrite=True)

    

def get_loss_function(dataset: Dataset):
    positive = 0
    negative = 0

    for i in range(len(dataset) // 2):
        _, label_sample1 = dataset.__getitem__(i)

        total = label_sample1.shape[0] * label_sample1.shape[1]
        
        sample_pos = label_sample1.sum() 

        positive += sample_pos
        negative += total - sample_pos

    if (positive == 0 or negative == 0):
        positive = 1.0
        negative = 1.0

    weight = torch.Tensor([negative/positive])

    return torch.nn.BCEWithLogitsLoss(pos_weight=weight)

def get_training_data(length: int = 20, label_scaling: int =1):
    psf = GuassionPSF(2.5)

    factor = 2
    options = TrainingDataSimulationOptions(
        grid_size=256 // factor,
        min_grains=2200 // (2 * factor * factor),
        max_grains=3200 // (2 * factor * factor),
        min_noise=0.05 ,
        max_noise=0.12,
        sample_rate=10,
        seconds=11, ## Decides how many frames each test samples has: total frames = sample_rate * seconds
        min_blinker_transition=0.04,
        max_blinker_transition=0.1,
        min_base_counts=6000,
        max_base_counts=12000,
        min_hole_chance=0.01,
        max_hole_chance=0.1,
        min_boundary_dimish=0.0,    
        max_boundary_dimish=1.0,
        min_blinker_strength=0.005,
        max_blinker_strength=0.08,
        min_blinkers_average=50,
        max_blinkers_average=90,
        psf=psf,
        label_scaling=label_scaling,
    )

    generated_dataset = GeneratedPLOutlineDataset(length=20, 
                                              sim_options=options, 
                                              transforms=transforms.Compose([
                                                BackgroundRemovalNormalize(),
                                                SkipFrames(skip=3),
                                              ]),
                                              empty_chance=0.05)

    return generated_dataset

def main():
    parser = argparse.ArgumentParser(description="Train a model using synthetic data")

    parser.add_argument('name', help='Name of model that will trained, this will be used in the filename for saving')
    
    parser.add_argument('--model',
                        type=Model,
                        choices=Model,
                        default=Model.DefaultSegmentation,
                        help='Model which will be trained')
    

    parser.add_argument('--batches', type=int, default=50, help='Amount of batches in each epoch step')
    parser.add_argument('--epochs', type=int, default=20, help='Amount of epochs the training will run for')
    
    parser.add_argument('--existing-model', type=str, required=False, help='Path of an existing model to use for training (optional)')

    parser.add_argument('--output-dir', '-o', type=str, default='./saved_models/', 
                        help='The path to which the model will be saved')
    
    args = parser.parse_args()

    print(f'Args: {args}')

    if args.model == Model.DefaultSegmentation:
        print(f'Training default segmentation model.')
        train_segmentation(args, PLSegmentationModel)
    elif args.model == Model.UnetSegmentation:
        print(f'Training Unet segmentation model')
        train_segmentation(args, PLSegmentationUnetModel)
    elif args.model == Model.UpscaleSegmentation:
        print(f'Training scaling segmentation model')
        train_scaling_segmentation_model(args, PLSegmentationScalingModel)
    elif args.model == Model.UnetUpscaleSegmentation:
        print(f'Training unet scaling segmentation model')
        train_scaling_segmentation_model(args, PLSegmentationUnetScalingModel)

if __name__ == '__main__':
    main()
