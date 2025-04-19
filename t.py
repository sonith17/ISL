import torch
import torch.nn as nn

x = torch.randn(1, 3, 63, 63)  # (batch_size, channels, height, width)
pool = nn.MaxPool2d(2, 2)  # kernel_size=2, stride=2
output = pool(x)

print("Input shape:", x.shape)   # (1, 3, 64, 64)
print("Output shape:", output.shape)  # (1, 3, 32, 32)
