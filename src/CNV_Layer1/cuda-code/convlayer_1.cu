#include "common.cuh"
#include "paramLib.h"
#include <iomanip>
#include <iostream>
#include <fstream>

//constant memory to store bias value
__constant__ float bias_value[4096];

//shared memory to store temporary data
extern __shared__ float SharedMemory[];

__global__ void ConvolutionLayer(float *input,
                                 float *filter, 
                                 float *output,
                                 TensorShape input_shape,
                                 TensorShape filter_shape,
                                 TensorShape output_shape,
                                 ConvArgs args,
                                 uint32_t bias_offset = 0,    //default
                                 uint32_t convNum_x = 1,      //default
                                 uint32_t convNum_y = 1,      //default
                                 uint32_t convMem_space = 128 //default
                                ){

    //different shared memory pointer
    size_t tile_size = blockDim.x * blockDim.y * blockDim.z;
    size_t filter_size = filter_shape.width * filter_shape.height * blockDim.z;

    float *sharedMem_input  = SharedMemory;   //memory szie -> blockDim.x * blockDim.y * blockDim.z * 4 Bytes
    float *sharedMem_filter = &SharedMemory[tile_size];  //memory size -> filter_shape.width * filter_shape.height * blockDim.z * 4 Bytes
    float *sharedMem_output = &SharedMemory[tile_size + filter_size];  //memory size -> convNum_x * convNum_y * convMem_space * 4 Bytes

    //global memory address index
    uint32_t global_input_offset, global_filter_offset, global_output_offset;

    //shared memory address index
    uint32_t shared_input_offset, shared_filter_offset, shared_output_offset;

    //thread mapping index to input original image
    uint32_t input_id_x = threadIdx.x + blockIdx.x * (convNum_x * args.stride_width);
    uint32_t input_id_y = threadIdx.y + blockIdx.y * (convNum_y * args.stride_height);
    uint32_t input_id_z = threadIdx.z;

    //define output map index for output shared memory
    uint32_t output_id_x, output_id_y, output_id_z;

    //operation on batch 1 (one of "n" input image)
    for(uint32_t n = 0; n < input_shape.counts; n++){

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        if(blockIdx.x * convNum_x * args.stride_width + filter_shape.width <= input_shape.width && blockIdx.y * convNum_y * args.stride_height + filter_shape.height <= input_shape.height){

            //calculate output shared memory index
            shared_output_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

            //initialize output shared memory to bias value for each filter
            if(shared_output_offset < convNum_x * convNum_y * convMem_space){

                sharedMem_output[shared_output_offset] = 0.0f;
            }
            __syncthreads();
                
            //calculate input shared memory index
            shared_input_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

            //load data from global input memory to shared input memory
            if((input_id_x < input_shape.width) && (input_id_y < input_shape.height)){

                //calculate global input memory address index to specific region
                global_input_offset = input_id_x + input_id_y * input_shape.width + input_shape.width * input_shape.height * (input_id_z + input_shape.channels * n);

                sharedMem_input[shared_input_offset] = input[global_input_offset];
            }
            __syncthreads();      
           
            //calculate filter shared memory index
            shared_filter_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

            //load data from global filter memory to shared filter memory
            if(shared_filter_offset < filter_size){

                //create global filter memory address index
                global_filter_offset = shared_filter_offset + filter_shape.width * filter_shape.height * filter_shape.channels * blockIdx.z;

                sharedMem_filter[shared_filter_offset] = filter[global_filter_offset];
            }
            __syncthreads();
                    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //convolution between filter and input
            for(uint32_t s = 0; s < convNum_y; s++){
                    
                for(uint32_t p = 0; p < convNum_x; p++){
                        
                    if(threadIdx.x < filter_shape.width && threadIdx.y < filter_shape.height){

                        //access shared input memory index
                        shared_input_offset = (args.stride_width * p + threadIdx.x) + (args.stride_height * s + threadIdx.y) * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
                            
                        //access shared filter memory index
                        shared_filter_offset = threadIdx.x + threadIdx.y * filter_shape.width + threadIdx.z * filter_shape.width * filter_shape.height;

                        //access shared output memory index
                        shared_output_offset = threadIdx.x + threadIdx.y * filter_shape.width + p * convMem_space + s * convNum_x * convMem_space;

                        //write convolution result to shared output memory
                        atomicAdd(&sharedMem_output[shared_output_offset], sharedMem_input[shared_input_offset] * sharedMem_filter[shared_filter_offset]);
                    }
                    __syncthreads();
                }
            } 

            //reduction
            uint32_t local_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

            for(uint32_t stride = 1; stride < convMem_space; stride *= 2){

                shared_output_offset = 2 * stride * local_id;

                if(shared_output_offset < convNum_x * convNum_y * convMem_space){

                    sharedMem_output[shared_output_offset] += sharedMem_output[shared_output_offset + stride];
                }

                __syncthreads();
            }

            //now the results of one thread block are stored in sharedMem_output[0], sharedMem_output[128], sharedMem_output[256], sharedMem_output[384]

                
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //mapping threads to global output
            output_id_x = threadIdx.x + blockIdx.x * convNum_x;
            output_id_y = threadIdx.y + blockIdx.y * convNum_y;
            output_id_z = blockIdx.z;

            if(threadIdx.x < convNum_x && threadIdx.y < convNum_y && output_id_x < output_shape.width && output_id_y < output_shape.height){

                //calculate global output memory address index
                global_output_offset = output_id_x + output_id_y * output_shape.width + output_id_z * output_shape.width * output_shape.height + n * output_shape.width * output_shape.height * output_shape.channels; 

                //re-calculate shared outpu memory access address
                shared_output_offset = (threadIdx.x + convNum_x * threadIdx.y) * convMem_space;

                //write result to global memory and add bias value
                output[global_output_offset] = sharedMem_output[shared_output_offset] + bias_value[blockIdx.z + bias_offset];

            }
            __syncthreads();
        }
    }
}

//calculate total number of elements in a NCHW tensor
size_t TensorSize (const TensorShape &t) {

	size_t size =  t.counts * t.channels * t.height * t.width;

	if (size == 0) {

		std::cout << "Invalid shape parameters" << std::endl;
	}
	return size;
}

void printArray(const float* array, size_t channels, size_t height, size_t width, size_t align){

    for(size_t n = 0; n < channels; ++n){

        std::cout << "Channel " << n << ":" << std::endl;

        for (size_t i = 0; i < height; ++i) {

            for(size_t j = 0; j < width; ++j){

                std::cout<< std::fixed << std::setprecision(5) << array[j + i * align + n * align * align];

                if (j < width - 1) {

                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << " " <<std::endl;
    }
}

//function to add paddings to input
float *addPadding(const float *original_input, size_t &total_elements, TensorShape &shape, int pad_height, int pad_width, float value = 0.0f){

    size_t padded_height = shape.height + 2 * pad_height;
    size_t padded_width  = shape.width + 2 * pad_width;

    //calculate total num of elements after padding
    total_elements = shape.counts * shape.channels * padded_height * padded_width;

    //allocate memory space for padded input
    float *padded_input = new float[total_elements];

    //initialize memory to pad value
    for(size_t i = 0; i < total_elements; ++i){

        padded_input[i] = value;
    }

    //map origianl input to correct position in NCHW layout
    for(size_t n = 0; n < shape.counts; ++n){

        for(size_t c = 0; c < shape.channels; ++c){

            for(size_t h = 0; h < shape.height; ++h){

                for(size_t w = 0; w < shape.width; ++w){

                    //calculate memory address
                    uint32_t original_index = w + h * shape.width + c * shape.height * shape.width + n * shape.channels * shape.height * shape.width;
                    uint32_t padded_index = (w + pad_width) + (h + pad_height) * padded_width + c * padded_height * padded_width + n * shape.channels * padded_height * padded_width;

                    padded_input[padded_index] = original_input[original_index];
                }
            }
        }
    }

    //free origianl input memory space
    delete[] original_input;

    return padded_input;
}

//functions to read source data and allocate host memory sapce
float *readBinary(const std::string &filepath, size_t &num_elements){

    //open binary file
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);

    if(!file.is_open()){

        std::cerr << "[Message]: Failed to open file: " << filepath << std::endl;
        return nullptr;
    }

    //obtain file size
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    //calculate element count
    num_elements = size / sizeof(float);

    //allocate memory space
    float *buffer = new float[num_elements];

    //read data from file to buffer
    if(!file.read(reinterpret_cast<char*>(buffer), size)){

        std::cerr << "[Message]: Error reading file: " << filepath << std::endl;

        //free memory space
        delete[] buffer;
        return nullptr;
    }

    return buffer;
}

//functions to write binary file
void writeBinary(const std::string &filename, float *content, size_t num_elements){

    //open file in write binary mode
    std::ofstream file(filename, std::ios::binary);

    if(!file){

        std::cerr << "[Message]: Unable to write file <" << filename << "> due to open error" << std::endl;
        return;
    }

    //write data from memory
    file.write(reinterpret_cast<const char*>(content), num_elements * sizeof(float));

    //finish
    file.close();
}

//comapre two float value
bool cmp_float(float a, float b, float epsilon){

    return std::fabs(a - b) < epsilon;
}

//calculate binary similarity
float compareBinary(const std::string &reference_filepath, const std::string &check_filepath, float epsilon){

    //open binary file in read mode
    std::ifstream ref_file(reference_filepath, std::ios::binary);
    std::ifstream chk_file(check_filepath, std::ios::binary);

    //check open status
    if(!ref_file.is_open()){

        std::cerr << "[Message]: Error opening reference file!" << std::endl;

        return -1;
    }

    if(!chk_file.is_open()){

        std::cerr << "[Message]: Error opening check file!" << std::endl;
        
        return -1;
    }

    //check file size
    std::ifstream::pos_type ref_fileSize = ref_file.tellg();
    std::ifstream::pos_type chk_fileSize = chk_file.tellg();

    if(ref_fileSize != chk_fileSize){

        std::cout << "[Message]: Files are in different size!" << std::endl;

        return -2;
    }

    float value1, value2;
    int closeCount = 0, totalCount = 0;

    //loop until end of file
    while(ref_file.read(reinterpret_cast<char *>(&value1), sizeof(value1)) && chk_file.read(reinterpret_cast<char *>(&value2), sizeof(value2))){

        if(cmp_float(value1, value2, epsilon)){

            ++closeCount;
        }

        ++totalCount;
    }

    //check if data is read
    if(totalCount == 0){

        std::cerr << "[Message]: No data has been read!" << std::endl;
        return -3;
    }

    //calculate similarity
    return static_cast<float>(closeCount)/totalCount;    
}

int main(){

    //setup
    Setup_GPU();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //tensor features
    uint32_t batchsize = 16;
    uint32_t bias_offset = 0;

    TensorShape input_shape = {batchsize, 3, 224, 224};
    TensorShape filter_shape = {64, 3, 11, 11};
    ConvArgs args = {2, 2, 4, 4, false};
    TensorShape output_shape;

    //NCHW layout
    output_shape.counts   = batchsize;
    output_shape.channels = filter_shape.counts;
    output_shape.height   = (input_shape.height + 2*args.pad_height - filter_shape.height)/args.stride_height + 1;
    output_shape.width    = (input_shape.width + 2*args.pad_width - filter_shape.width)/args.stride_width + 1;


    //source binary file filepath * CHANGE THE FILEPATH TO YOURS *
    std::string input_filepath  = "../pytorch-code/conv_input_tensor.bin";
    std::string filter_filepath = "../pytorch-code/conv_filter_tensor.bin";
    std::string bias_filepath   = "../pytorch-code/conv_bias_value.bin";

    size_t input_num_elements;
    size_t padded_input_num_elements;
    size_t filter_num_elements;
    size_t bias_num_elements;
    size_t output_num_elements = TensorSize(output_shape);

    std::cout << "                                                                                                   " << std::endl;
    std::cout << "                                                                                                   " << std::endl;
    std::cout << "                                                                                                   " << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "                  Start reading binary files and allocating host memory space                      " << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "                                                                                                   " << std::endl;

    //create host input pointer and read input binary file
    float *in = readBinary(input_filepath, input_num_elements);

    if(in == nullptr){

        std::cerr << "[Message]: Failed to allocate host original input memory space!" << std::endl;
    }

    //add paddings to input, re-map the input memory space
    float *CNV_layer_in = addPadding(in, padded_input_num_elements, input_shape, args.pad_height, args.pad_width);

    if(CNV_layer_in == nullptr){

        std::cerr << "[Message]: Failed to allocate host padded input memory space!" << std::endl;
    }

    //update padded input shape
    input_shape.width += 2 * args.pad_width;
    input_shape.height += 2 * args.pad_height;

    float input_hostMem_space = padded_input_num_elements * sizeof(float) / (1024 * 1024);

    std::cout << "[Message]: host memory space required by input after padding is: " << input_hostMem_space << " MB" << std::endl;

    //create host filter pointer and read filter binary file
    float *CNV_layer_filter = readBinary(filter_filepath, filter_num_elements);

    if(CNV_layer_filter == nullptr){

        std::cerr << "Failed to allocate host filter memory space!" << std::endl;
    }

    float filter_hostMem_space = filter_num_elements * sizeof(float) / 1024 ;

    std::cout << "[Message]: host memory space required by weights is: " << filter_hostMem_space << " KB" << std::endl;

    //create host bias value pointer and read bias
    float *CNV_layer_bias   = readBinary(bias_filepath, bias_num_elements);

    if(CNV_layer_bias == nullptr){

        std::cerr << "[Message]: Failed to allocate bias memory space!" << std::endl;
    }

    float bias_hostMem_space = bias_num_elements * sizeof(float);

    std::cout << "[Message]: host memory space required by bias values is: " << bias_hostMem_space << " Bytes" << std::endl;

    //allocate host output memory space
    float *CNV_layer_output = new float[output_num_elements];

    float output_hostMem_space = output_num_elements * sizeof(float) / (1024 * 1024);

    std::cout << "[Message]: host memory space required by output is: " << output_hostMem_space << " MB" << std::endl;
    std::cout << "[Message]: host memory allocation complete!" << std::endl;
    
    std::cout << "                                                                                                   " << std::endl;
    std::cout << "                                                                                                   " << std::endl;
    std::cout << "                                                                                                   " << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "                  Start allocating device memory space and transferring data                       " << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "                                                                                                   " << std::endl;

    //device memory pointer
    float *input_device, *filter_device, *output_device;

    //allocate device memory space and transfer data from host to device
    CUDA_CHECK(cudaMalloc(&input_device, padded_input_num_elements*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_device, CNV_layer_in, padded_input_num_elements*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&filter_device, filter_num_elements*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(filter_device, CNV_layer_filter, TensorSize(filter_shape)*sizeof(float), cudaMemcpyHostToDevice));

    //move data from device global memory to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(bias_value, CNV_layer_bias, bias_num_elements*sizeof(float), bias_offset*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&output_device, TensorSize(output_shape)*sizeof(float)));

    std::cout << "[Message]: device memory allocation complete!" << std::endl;
    std::cout << "[Message]: transferring data from host memory to GPU memory..." << std::endl;

    std::cout << "                                                                                                   " << std::endl;
    std::cout << "                                                                                                   " << std::endl;
    std::cout << "                                                                                                   " << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "                                  Creating grids and thread blocks                                 " << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "                                                                                                   " << std::endl;

    size_t BLOCK_SIZE = 16;
    size_t block_dim_x = BLOCK_SIZE;
    size_t block_dim_y = BLOCK_SIZE;
    size_t block_dim_z = input_shape.channels;

    size_t convNum_x = (BLOCK_SIZE - filter_shape.width + args.stride_width) / args.stride_width;
    size_t convNum_y = (BLOCK_SIZE - filter_shape.height + args.stride_height) / args.stride_height;
    size_t convMem_space = 32 * ceil((float)(filter_shape.width * filter_shape.height)/(float)(32));

    size_t grid_dim_x = ceil((float)(input_shape.width)/(float)(convNum_x * args.stride_width));
    size_t grid_dim_y = ceil((float)(input_shape.height)/(float)(convNum_y * args.stride_height));
    size_t grid_dim_z = filter_shape.counts;

    size_t sharedMemory_size = ((BLOCK_SIZE * BLOCK_SIZE + filter_shape.width * filter_shape.height) * block_dim_z + convNum_x * convNum_y * convMem_space) * sizeof(float);

    std::cout << "[Message]: block dimension is defined as (" << block_dim_x << ", " << block_dim_y << ", " << block_dim_z << ")" << std::endl;
    std::cout << "[Message]: grid dimension is defined as (" << grid_dim_x << ", " << grid_dim_y << ", " << grid_dim_z << ")" << std::endl;
    std::cout << "[Message]: numbers of convolution operation on X direction in a thread block: " << convNum_x << std::endl;
    std::cout << "[Message]: numbers of convolution operation on Y direction in a thread block: " << convNum_y << std::endl;
    std::cout << "[Message]: block shared memory size is defined as " << (float)(sharedMemory_size/1024)  << " KB" << std::endl;

    std::cout << "                                                                                                   " << std::endl;
    std::cout << "                                                                                                   " << std::endl;
    std::cout << "                                                                                                   " << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "                                        Launching kernel                                           " << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "                                                                                                   " << std::endl;

    cudaEventRecord(start);

    dim3 block(block_dim_x, block_dim_y, block_dim_z);
    dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);

    ConvolutionLayer <<<grid, block, sharedMemory_size>>> (input_device, filter_device, output_device, input_shape, filter_shape, output_shape, args, bias_offset, convNum_x, convNum_y, convMem_space);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    std::cout << "[Message]: kernel execution complete!"<< std::endl;

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "[Message]: kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "                                                                                                   " << std::endl;
    std::cout << "                                                                                                   " << std::endl;
    std::cout << "                                                                                                   " << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "                                       Verifying the results                                       " << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "                                                                                                   " << std::endl;

    std::cout << "[Message]: transferring data from GPU memory to host memory..." << std::endl;
    //copy result from device memory to host memory
    CUDA_CHECK(cudaMemcpy(CNV_layer_output, output_device, TensorSize(output_shape)*sizeof(float), cudaMemcpyDeviceToHost));

    //output file and reference file path
    std::string chk_filepath  = "conv_output_check.bin";
    std::string ref_filepath = "../pytorch-code/conv_output_reference.bin";

    writeBinary(chk_filepath, CNV_layer_output, output_num_elements);

    std::cout << "[Message]: the similarity between output and reference is: " << compareBinary(ref_filepath, chk_filepath, 1e-5) * 100.f << "%" << std::endl;

    //update bias memory offset
    bias_offset += filter_shape.counts;

    //free device and host memory
    CUDA_CHECK(cudaFree(input_device));
    CUDA_CHECK(cudaFree(filter_device));
    CUDA_CHECK(cudaFree(output_device));
    delete[] CNV_layer_in;
    delete[] CNV_layer_filter;
    delete[] CNV_layer_bias;
    delete[] CNV_layer_output;

    //calculate throughput
    uint32_t multiplyFLOPs_per_outputElement = input_shape.channels * filter_shape.height * filter_shape.width;
    uint32_t addFLOPs_per_outputElement = multiplyFLOPs_per_outputElement - 1;
    uint32_t total_FLOPs_per_outputElement = multiplyFLOPs_per_outputElement + addFLOPs_per_outputElement;

    uint32_t total_FLOPs = TensorSize(output_shape) * total_FLOPs_per_outputElement;

    float seconds = milliseconds / 1000;
    float throughput = total_FLOPs / (seconds * (1000 * 1000 * 1000));

    std::cout << "[Message]: throughput is: " << throughput << " GFLOPs/s" << std::endl;

    return 0;
}
