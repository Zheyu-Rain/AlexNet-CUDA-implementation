import torch
import numpy as np
import torchvision.models as models
from torchvision.models import AlexNet_Weights

# load pretrained AlexNet model
layer = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

# extract first convolution layer, pre-trained weights and bias values
first_conv_layer = layer.features[0]
weights = layer.features[0].weight.data
bias = layer.features[0].bias.data

# transform pretrained weights and bias to numpy format
weights_np = weights.numpy()
bias_np = bias.numpy()

# write binary file
with open('conv_filter_tensor.bin', 'wb') as f:
    f.write(weights_np.tobytes())
with open('conv_bias_value.bin', 'wb') as f:
    f.write(bias_np.tobytes())

# disable weight change
for param in first_conv_layer.parameters():
    param.requires_grad = False

# move layer to gpu
first_conv_layer = first_conv_layer.cuda()

# create input
torch.manual_seed(5)
input_tensor = torch.randn(16, 3, 224, 224)

# transform input tensor to numpy format
input_tensor_np = input_tensor.numpy()

# write binary file
with open('conv_input_tensor.bin', 'wb') as f:
    f.write(input_tensor_np.tobytes())

# move input to gpu
input_tensor_cuda = input_tensor.cuda()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

# forward propagation
output = first_conv_layer(input_tensor_cuda)

end.record()
torch.cuda.synchronize()

# calculate layer execution time
elapsed_time_ms = start.elapsed_time(end)

# move data from gpu to cpu
output_cpu = output.cpu()

# transform output data to numpy format
output_np = output_cpu.numpy()

# write binary file
with open('conv_output_reference.bin', 'wb') as f:
    f.write(output_np.tobytes())

print(f"Convolution operation time: {elapsed_time_ms} ms")
