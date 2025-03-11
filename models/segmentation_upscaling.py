import torch
from models.conv_lstm import ConvLSTM
from models.saveable_model import SaveableModel
from models.unet.unet_model import UNet
import math

from models.unet.unet_parts import UpScaleDoubleConv

class PLSegmentationScalingModel(SaveableModel):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, num_layers: int=1, scaling: int=2) -> None:
        super().__init__()
    
        assert(math.log2(scaling).is_integer(), "Scaling should be a power of two")
        assert(hidden_dim % scaling == 0, "Scaling should devide hidden_dim evenly")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.scaling = scaling

        self.convLSTM = ConvLSTM(input_dim=input_dim, 
                                hidden_dim=hidden_dim, 
                                kernel_size=(kernel_size, kernel_size), 
                                num_layers=num_layers, 
                                batch_first=True)
        
        self.upscaling_layer = UpScaleDoubleConv(hidden_dim, 1, bilinear=False)
        


    def forward(self, input, inference: bool = False):
        output_layers, _ = self.convLSTM.forward(input, None)

        last_hidden_layer = output_layers[0][:, -1, :, :, :]
        output = self.upscaling_layer(last_hidden_layer)

        if inference:
            return  torch.nn.Sigmoid()(output)
        else:
            return output

    def get_config(self) -> dict:
        result = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "kernel_size": self.kernel_size,
            "num_layers": self.num_layers,
            "scaling": self.scaling,
        }

        return result

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)
