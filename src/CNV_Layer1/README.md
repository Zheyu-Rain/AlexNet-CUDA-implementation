Before using the code, make sure that you have properly installed the CUDA environment as well as the Pytorch environment. 

This folder contains two implementations of AlexNet's first convolutional layer, which are located in the <cuda-code> and <pytorch-code> respectively.

The implementation in pytorch-code folder is extracted from a pre-trained AlexNet model. Please first run pytorch version to obtain input tensor, output tensor and pre-trained weights and bias values in binary format.

Then you can compile cuda code by command: nvcc -arch=compute_xx -code=sm_xx conv
