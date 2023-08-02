import torch
import torch.nn as nn
import torch.nn.functional as F
class GaussianFourierFeatureTransform(torch.nn.Module):
    def __init__(self, num_input_channels, mapping_size=93, scale=25):
        super().__init__()
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale)
    def forward(self, x):
        x = x.squeeze(0)
        x = torch.sin(x @ self._B.to(x.device))
        return x

class Nerf_positional_embedding(torch.nn.Module):
    def __init__(self, multires):
        super().__init__()
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.brand_number = multires

    def forward(self, x):
        x = x.squeeze(0)
        freq_bands = 2.**torch.linspace(0, self.brand_number-1, self.brand_number)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


class DenseLayer(nn.Linear):
    def __init__(self, in_dim, out_dim, activation = "relu"):
        self.activation = activation
        super().__init__(in_dim, out_dim,)
    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class No_positional_encoding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.squeeze(0)
        return x