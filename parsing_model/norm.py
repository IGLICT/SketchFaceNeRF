import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch
class SpecificNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(SpecificNorm, self).__init__()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.mean = torch.from_numpy(self.mean).float().cuda()
        self.mean = self.mean.view([1, 3, 1, 1])

        self.std = np.array([0.229, 0.224, 0.225])
        self.std = torch.from_numpy(self.std).float().cuda()
        self.std = self.std.view([1, 3, 1, 1])

    def forward(self, x):
        mean = self.mean.expand([1, 3, x.shape[2], x.shape[3]])
        std = self.std.expand([1, 3, x.shape[2], x.shape[3]])

        x = (x - mean) / std

        return x

class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask