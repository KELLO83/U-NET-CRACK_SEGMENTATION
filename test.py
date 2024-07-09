import torch
import torch.nn as nn

# Define the ConvTranspose2d layer
conv_transpose = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)

# Example input tensor of shape (batch_size, channels, height, width)
input_tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

# Apply the transposed convolution
output_tensor = conv_transpose(input_tensor)

print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)
print("Output Tensor:", output_tensor)

print(conv_transpose.weight)
