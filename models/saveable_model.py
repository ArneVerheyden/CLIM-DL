import torch
from abc import ABC, abstractmethod
import os
from pathlib import Path

from typing import TypeVar

T = TypeVar('T', bound='SaveableModel')

class SaveableModel(torch.nn.Module, ABC):
    @abstractmethod
    def get_config(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_config(cls: type[T], config: dict) -> T:
        pass

    def save(self, save_path: str, overwrite=False):
        path = Path(save_path)
        if path.is_dir():
            raise ValueError(f"Path '{save_path}' is a directory.")
        if path.is_file() and not overwrite:
            raise ValueError(f"File '{save_path}' already exists, overwrite is set to False.")
        
        path.parent.mkdir(parents=True, exist_ok=True)

        config = self.get_config()
        config['model_type'] = self.__class__.__name__

        save_dict = {
            'config': config,
            'state_dict': self.state_dict(),
        }

        torch.save(save_dict, path)
        print(f"Saved model to file: '{path}'")

    @classmethod
    def load(cls, load_path: str):
        path = Path(load_path) 
        if not path.is_file():
            raise ValueError(f"No model file found at path: '{load_path}'")

        save_dict = torch.load(path, weights_only=True)
        config: dict = save_dict['config']

        if config.pop('model_type', None) != cls.__name__:
            raise ValueError(
                f"Atteming to load a model of the wrong type. Expected: '{cls.__name__}', got: '{config['model_type']}'"
            )       
        
        model = cls.from_config(config)
        model.load_state_dict(save_dict['state_dict'])

        print(f"Model loaded from: '{path}'")

        return model

        
