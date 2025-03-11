import torch

def normalize_data(data: torch.Tensor, axis: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if (data.dtype != dtype):
        data = data.to(dtype)

    result = data - data.mean(axis=axis, keepdim=True)
    result /= result.std(axis=axis, keepdim=True)

    return result