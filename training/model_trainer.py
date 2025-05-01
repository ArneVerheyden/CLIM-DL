from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod 

import torch
from models.saveable_model import SaveableModel

import gc

class ModelTrainer(ABC):
    def __init__(self, 
                 model: SaveableModel, 
                 training_data: Dataset, 
                 testing_data: Dataset, 
                 learning_rate: float, 
                 loss_function):
        parameters = model.parameters()

        self.model = model
        self.training_data = training_data
        self.testing_data = testing_data
        self.optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        self.loss_function = loss_function
        
        self.epoch_losses = []
        self.all_losses = []

        self.training = False

    @abstractmethod
    def step(self, input: torch.Tensor, labels: torch.Tensor) -> float:
        pass

    def train(self, epochs: int):
        self.training = True

        model = self.model
        model.train()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=4, verbose=True,
            min_lr=1**-6
        )

        training_loader = DataLoader(self.training_data, batch_size=1, shuffle=False)

        for i in range(epochs):
            epoch_loss = 0.0

            for input, labels in training_loader:
                loss = self.step(input, labels)

                loss_value = loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                del input, labels, loss,
                gc.collect()

                epoch_loss += loss_value
                self.all_losses.append(loss_value)

            av_loss = epoch_loss / len(training_loader)
            print(f'Epoch {i + 1} done, average loss: {av_loss}. {float(i + 1) / float(epochs) * 100} % done.')

            scheduler.step(av_loss)
            self.epoch_losses.append(av_loss)

            if not self.training:
                return

        self.training = False

    def save(self, path: str, overwrite=False):
        self.model.save(path, overwrite)
    
    def stop(self):
        self.training = False

