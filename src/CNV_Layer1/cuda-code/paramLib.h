#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <vector>
#include <algorithm>

//declare a constructor to store tensor features
typedef struct TensorShape_t
{
    uint32_t counts;    
    uint32_t channels;     
    uint32_t width;   
    uint32_t height;   

} TensorShape;

//declare a constructor to store convolution arguments
typedef struct ConvArgs_t
{
    uint32_t pad_width;
    uint32_t pad_height;
    uint32_t stride_width;
    uint32_t stride_height;

    bool activation;
} ConvArgs;

//declare a constructor to store pooling arguments
typedef struct PoolArgs_t
{
    uint32_t pad_width;
    uint32_t pad_height;
    uint32_t stride_width;
    uint32_t stride_height;

} PoolArgs;

