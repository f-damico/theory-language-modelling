import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, stride=1, bias=False):
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.filter = nn.Parameter(torch.randn(out_channels, in_channels, filter_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.conv1d(x, self.filter, self.bias, stride=self.stride) / (
            self.filter.size(1) * self.filter.size(2)
        ) ** 0.5


class hCNN(nn.Module):
    def __init__(self, input_dim, patch_size, in_channels, nn_dim, out_channels, num_layers, bias=False, norm='std'):
        super().__init__()

        receptive_field = patch_size ** num_layers
        assert input_dim % receptive_field == 0, 'patch_size**num_layers must divide input_dim!'

        self.hidden = nn.Sequential(
            nn.Sequential(MyConv1d(in_channels, nn_dim, patch_size, stride=patch_size, bias=bias), nn.ReLU()),
            *[
                nn.Sequential(MyConv1d(nn_dim, nn_dim, patch_size, stride=patch_size, bias=bias), nn.ReLU())
                for _ in range(1, num_layers)
            ],
        )
        self.readout = nn.Parameter(torch.randn(nn_dim, out_channels))
        if norm == 'std':
            self.norm = nn_dim ** 0.5
        elif norm == 'mf':
            self.norm = nn_dim
        else:
            raise ValueError("norm must be 'std' or 'mf'")

    def forward(self, x):
        x = self.hidden(x)
        x = x.mean(dim=[-1])
        x = x @ self.readout / self.norm
        return x

    def _all_weight_matrices(self):
        for block in self.hidden:
            conv = block[0]
            yield conv.filter
        yield self.readout.t()

    @staticmethod
    def _reshape_weight(weight):
        if weight.ndim > 2:
            return weight.reshape(weight.size(0), -1)
        return weight

    @staticmethod
    def _spectral_norm(weight):
        w = hCNN._reshape_weight(weight.detach().to(torch.float64))
        return torch.linalg.svdvals(w)[0]

    @staticmethod
    def _two_one_norm(weight):
        w = hCNN._reshape_weight(weight.detach().to(torch.float64))
        return torch.norm(w, p=2, dim=0).sum()

    @torch.no_grad()
    def compute_model_norm(self):
        weights = list(self._all_weight_matrices())
        if len(weights) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        s_list = [self._spectral_norm(weight) for weight in weights]
        t_list = [self._two_one_norm(weight) for weight in weights]

        prod_spec = torch.prod(torch.stack(s_list))
        correction_sum = torch.tensor(0.0, dtype=torch.float64, device=prod_spec.device)
        eps = torch.tensor(1e-12, dtype=torch.float64, device=prod_spec.device)

        for i in range(len(weights)):
            if i < len(weights) - 1:
                t_sum = torch.stack(t_list[i + 1:]).sum()
            else:
                t_sum = torch.tensor(0.0, dtype=torch.float64, device=prod_spec.device)
            correction_sum = correction_sum + (t_sum ** (2.0 / 3.0)) / (s_list[i] ** (2.0 / 3.0) + eps)

        norm_value = prod_spec * (correction_sum ** (3.0 / 2.0))
        return norm_value.to(dtype=next(self.parameters()).dtype)

    @torch.no_grad()
    def compute_l2_norm(self):
        total_sq = torch.tensor(0.0, dtype=torch.float64, device=next(self.parameters()).device)
        for weight in self._all_weight_matrices():
            total_sq += torch.sum(weight.detach().to(torch.float64) ** 2)
        return torch.sqrt(total_sq).to(dtype=next(self.parameters()).dtype)
