import torch
from models.conv_lstm import ConvLSTM
from models.saveable_model import SaveableModel

class PLSegmentationModelV2(SaveableModel):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, num_layers: int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.convLSTM = ConvLSTM(input_dim=input_dim, 
                                hidden_dim=hidden_dim, 
                                kernel_size=(kernel_size, kernel_size), 
                                num_layers=num_layers, 
                                batch_first=True)
        
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=hidden_dim, 
                                        out_channels=hidden_dim,
                                        kernel_size=kernel_size,
                                        padding="same"),
            torch.nn.ELU(),
            torch.nn.Conv2d(in_channels=hidden_dim, 
                                        out_channels=hidden_dim,
                                        kernel_size=kernel_size,
                                        padding="same"),    
            torch.nn.ELU(),        
            torch.nn.Conv2d(in_channels=hidden_dim, 
                                        out_channels=1,
                                        kernel_size=kernel_size,
                                        padding="same"),
        )



    def forward(self, input: torch.Tensor, inference: bool = False) -> torch.Tensor:
        output_layers, _ = self.convLSTM.forward(input, None)

        last_hidden_layer = output_layers[0][:, -1, :, :, :]

        output = self.convs(last_hidden_layer)

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
        }

        return result

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)
