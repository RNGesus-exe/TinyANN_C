#ifndef CNN_H
#define CNN_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*
Format for using TinyANN Library
Input Layer CHANNELS ROWS COLS (We need to resize image using BILINEAR_INTERPOLATION)
PreProcessing NORMALIZE(1,0.5,0.5)
MAX_WEIGHTS || MAX_BIAS = 2e6
Hidden Layers 'N' $
1 C STRIDE(2) PADDING(2) KERNEL_SIZE(3) ACTIVATION(R) END_OF_LINE($) IN_DIM() OUT_DIM()
2 M 0 0 2 $
3 C 0 0 3 R $
4 F $
.
.
N N R $
Output Layer ARG_MAX

If you don't want to add an attribute with a layer just write '0'
CONV2D == C(1) && MAXPOOL = M(2) && FLATTEN == F(3) && FULLY_CONNECTED == N(4)
ACTIVATON : RELU(1)
It is important to flatten the feature_map before sending to FULLY_CONNECTED layer

E.g.
3 128 128
9
1 2 1 3 1 3 16
2 2 0 2 0 0 0
1 2 1 3 1 16 32
2 2 0 2 0 0 0
1 1 1 3 1 32 64
2 2 0 2 0 0 0
3 0 0 0 0 0 0
4 0 0 0 0 1024 100
4 0 0 0 0 100 50
4 0 0 0 0 50 10

*/

// ERROR CODES
#define SUCCESS 0
#define MEMORY_ALLOCATION_FAILED -1
#define MEMORY_REGION_EXCEEDED -2
#define FILE_NOT_READABLE -3

// Constants
#define MAX_LAYER_INFO_SIZE 7
#define MAX_LINE_SIZE 100

enum HiddenLayerAttribute { _operation, _stride, _padding, _kernel_size, _activation, _input, _output };

enum Operations { _convolution = 1, _maxpool, _flatten, _fully_connected };

typedef struct MemoryRegion {
    size_t size;
    float* memory_start;
    float* memory_used;
} MemoryRegion;

typedef struct Tensor {
    size_t info[MAX_LAYER_INFO_SIZE];
    size_t channels;
    size_t height;
    size_t width;
    float* weight_start;
    float* weight_end;
    float* bias_start;
    float* bias_end;
    float* start;
    float* end;
} Tensor;

typedef struct TinyANN {
    size_t total_layers;
    MemoryRegion memory_block;
    Tensor* tensors;
    size_t image_filters;
    size_t image_rows;
    size_t image_cols;
} TinyANN;

//=====Memory Region====
int allocateMemoryRegion(TinyANN* tinyANN);

int createTensors(TinyANN* tinyANN);

int deallocateMemoryRegion(TinyANN* tinyANN);

//=====Neural Network=====

int initNetwork(TinyANN* tinyANN, const char* network_config_path, const char* param_path);

int loadParams(TinyANN* tinyANN, const char* param_path);

void inference(TinyANN* tinyANN, float* image);

int destroyNetwork(TinyANN* tinyANN);

#endif // CNN_H
