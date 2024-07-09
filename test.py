import torch
import torch.nn as nn

# Define the ConvTranspose2d layer
conv_transpose = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)

# Example input tensor of shape (batch_size, channels, height, width)
input_tensor = torch.randn(size=(1,6,6))

# Apply the transposed convolution
output_tensor = conv_transpose(input_tensor)

print("Input Tensor Shape:", input_tensor.shape) # 1 6 6
print("Output Tensor:", output_tensor)
print("Output Tensor Shape:", output_tensor.shape) # 1 12 12

print("convTranspose2d Weight" , conv_transpose.weight) # 3 3
