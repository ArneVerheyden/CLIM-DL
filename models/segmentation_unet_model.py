import torch
from models.conv_lstm import ConvLSTM
from models.saveable_model import SaveableModel
from models.unet.unet_model import UNet

class PLSegmentationUnetModel(SaveableModel):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, num_layers: int=1, n_start_unet_channels=64) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.n_start_unet_channels = n_start_unet_channels

        self.convLSTM = ConvLSTM(input_dim=input_dim, 
                                hidden_dim=hidden_dim, 
                                kernel_size=(kernel_size, kernel_size), 
                                num_layers=num_layers, 
                                batch_first=True)
        
        self.unet = UNet(hidden_dim, 1, n_start_channels=n_start_unet_channels)


    def forward(self, input, inference: bool = False):
        output_layers, _ = self.convLSTM.forward(input, None)

        last_hidden_layer = output_layers[0][:, -1, :, :, :]
        output = self.unet.forward(last_hidden_layer)

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
            "n_start_unet_channels": self.n_start_unet_channels,
        }

        return result

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)
