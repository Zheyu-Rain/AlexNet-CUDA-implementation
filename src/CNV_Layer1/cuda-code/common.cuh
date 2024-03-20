#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#define CUDA_CHECK(call)                __cudaCheck(call, __FILE__, __LINE__)
#define KERNEL_CHECK(call)              __kernelCheck(__FILE__, __LINE__)

static void __cudaCheck(cudaError_t error_code, const char *FileName, const int LineNumber)
{
    if(error_code != cudaSuccess)
    {
        std::cout << "CUDA error occurs in file: " << FileName << "\r\n"
          << "Line: " << LineNumber << ".\r\n"
          << "Code = " << error_code << "\r\n"
          << "Error Name: " << cudaGetErrorName(error_code) << "\r\n"
          << "Description: " << cudaGetErrorString(error_code) << "\r\n";

        exit(1);
    }
}

static void __kernelCheck(const char * FileName, const int LineNumber)
{
    cudaError_t error = cudaPeekAtLastError();
    
    if(error != cudaSuccess)
    {
        std::cout << "Kernel check found CUDA error occurs in file: " << FileName << "\r\n"
            << "Line: " << LineNumber << "\r\n"
            << "Code: " << cudaGetErrorName(error) << "\r\n"
            << "Description: " << cudaGetErrorString(error) << "\r\n";
        
        exit(1);
    }
}

void Setup_GPU()
{

    //Check the available GPUs in the system using CUDA runtime API
    int DeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&DeviceCount);

    if(error != cudaSuccess || DeviceCount == 0)
    {
        std::cout <<"SETUP INFO: No CUDA Device Found!" << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "SETUP INFO: Found <" << DeviceCount << "> CUDA enabled GPU" << std::endl;
    
    }

    //Setup the execution GPU
    int Device = 0;
    cudaDeviceProp Props;
    CUDA_CHECK(cudaGetDeviceProperties(&Props, Device));
    
    std::cout << "SETUP INFO: Setup computing on device: " << Props.name << std::endl;
    std::cout << "SETUP INFO: Device compute capability: " << Props.major << "." << Props.minor << std::endl;
    std::cout << "SETUP INFO: Device global memory size: " << (Props.totalGlobalMem / (1024 * 1024 * 1024)) + 1 << " GB" << std::endl;
    std::cout << "SETUP INFO: Device global memory bit width: " << Props.memoryBusWidth << " Bit" << std::endl;
    std::cout << "SETUP INFO: Device streaming multiprocessors count: " << Props.multiProcessorCount << " SMs" << std::endl;
    std::cout << "SETUP INFO: Device avaliable constant memory size: " << Props.totalConstMem / 1024 << " KB" << std::endl;
    std::cout << "SETUP INFO: Device L2 cache size: " << Props.l2CacheSize / (1024 * 1024) << " MB" << std::endl;
    std::cout << "" << std::endl;
  

    std::cout << "SETUP INFO: Maximum size of avaliable shared memory per thread block: " << Props.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "SETUP INFO: Maximum number of avaliable 32-bit register per thread block: " << Props.regsPerBlock /1024 << " K" << std::endl;
    std::cout << "SETUP INFO: Maximum size of avaliable shared memory per SM: " << Props.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << "SETUP INFO: Maximum number of avaliable 32-bit register per SM: " << Props.regsPerMultiprocessor /1024 << " K" << std::endl;
    std::cout << "SETUP INFO: Maximum number of active threads per SM: " << Props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "SETUP INFO: Maximum number of active blocks per SM: " << Props.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "" << std::endl;
}