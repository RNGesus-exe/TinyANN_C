#include "../include/cnn.h"

int initNetwork(TinyANN* tinyANN, const char* network_config_path, const char* param_path) {
    FILE* network_config = fopen(network_config_path, "rb");

    if (network_config == NULL) {
        fprintf(stderr, "ERROR TINY_ANN: File (%s) does not exist\n", network_config_path);
        return FILE_NOT_READABLE;
    }

    fscanf(network_config, "%ld %ld %ld", &tinyANN->image_filters, &tinyANN->image_rows, &tinyANN->image_cols);

    fscanf(network_config, "%ld", &tinyANN->total_layers);

    tinyANN->tensors = (Tensor*)malloc(tinyANN->total_layers * sizeof(Tensor));

    for (int i = 0; i < tinyANN->total_layers; i++) {
        for (int j = 0; j < MAX_LAYER_INFO_SIZE; j++) {
            fscanf(network_config, "%ld", &tinyANN->tensors[i].info[j]);
        }
    }

    fclose(network_config);

    allocateMemoryRegion(tinyANN);

    createTensors(tinyANN);

    loadParams(tinyANN, param_path);

    for (int i = 0; i < tinyANN->total_layers; i++) {
        printf("layer %d : ", i + 1);
        for (int j = 0; j < MAX_LAYER_INFO_SIZE; j++) {
            printf("%ld ", tinyANN->tensors[i].info[j]);
        }
        printf("%ld %ld %ld \n", tinyANN->tensors[i].channels, tinyANN->tensors[i].height, tinyANN->tensors[i].width);
    }

    return SUCCESS;
}

int loadParams(TinyANN* tinyANN, const char* param_path) {

    for (int i = 0; i < tinyANN->total_layers; i++) {
        if (tinyANN->tensors[i].info[_operation] == _convolution || tinyANN->tensors[i].info[_operation] == _fully_connected) {
            tinyANN->tensors[i].weight_start = tinyANN->memory_block.memory_used;
            tinyANN->tensors[i].weight_end = tinyANN->memory_block.memory_used +=
                (tinyANN->tensors[i].info[_input] * tinyANN->tensors[i].info[_output] * tinyANN->tensors[i].info[_kernel_size] *
                 tinyANN->tensors[i].info[_kernel_size]);
            tinyANN->tensors[i].bias_start = tinyANN->memory_block.memory_used;
            tinyANN->tensors[i].bias_end = tinyANN->memory_block.memory_used += tinyANN->tensors[i].info[_output];
        } else {
            tinyANN->tensors[i].weight_start = tinyANN->tensors[i].bias_start = tinyANN->tensors[i].bias_end =
                tinyANN->tensors[i].weight_end = NULL;
        }
    }

    FILE* param_config = fopen(param_path, "rb");

    if (param_config == NULL) {
        fprintf(stderr, "ERROR TINY_ANN: File (%s) does not exist\n", param_path);
    }

    for (int t = 0; t < tinyANN->total_layers; t++) {
        if (tinyANN->tensors[t].info[_operation] != _convolution && tinyANN->tensors[t].info[_operation] != _fully_connected)
            continue;

        int ind = 0;
        for (int out = 0; out < tinyANN->tensors[t].info[_output]; out++) {
            for (int in = 0; in < tinyANN->tensors[t].info[_input]; in++) {
                for (int r = 0; r < tinyANN->tensors[t].info[_kernel_size]; r++) {
                    for (int c = 0; c < tinyANN->tensors[t].info[_kernel_size]; c++) {
                        fscanf(param_config, "%f", &tinyANN->tensors[t].weight_start[ind]);
                        ind++;
                    }
                }
            }
        }
        ind = 0;
        for (int out = 0; out < tinyANN->tensors[t].info[_output]; out++) {
            fscanf(param_config, "%f", &tinyANN->tensors[t].bias_start[out]);
            ind++;
        }
    }

    fclose(param_config);

    return SUCCESS;
}

void inference(TinyANN& tinyANN, float* image) {
    return;
}

int createTensors(TinyANN* tinyANN) {

    for (int i = 0; i < tinyANN->total_layers; i++) {
        tinyANN->tensors[i].start = tinyANN->memory_block.memory_used;
        tinyANN->tensors[i].end = tinyANN->memory_block.memory_used +=
            (tinyANN->tensors[i].channels * (tinyANN->tensors[i].width + 2 * tinyANN->tensors[i].info[_padding]) *
             (tinyANN->tensors[i].height + 2 * tinyANN->tensors[i].info[_padding]));
    }

    return SUCCESS;
}

int destroyNetwork(TinyANN* tinyANN) {
    if (tinyANN->tensors) {
        free(tinyANN->tensors);
        tinyANN->tensors = NULL;
    }

    return deallocateMemoryRegion(tinyANN);
}

int allocateMemoryRegion(TinyANN* tinyANN) {
    size_t memory_size = 0;
    size_t height = tinyANN->image_rows;
    size_t width = tinyANN->image_cols;

    // Memory for feature maps
    for (int i = 0; i < tinyANN->total_layers; i++) {

        tinyANN->tensors[i].height = height;
        tinyANN->tensors[i].width = width;
        tinyANN->tensors[i].channels = tinyANN->tensors[i].info[_input];

        if (tinyANN->tensors[i].info[_operation] == _flatten || tinyANN->tensors[i].info[_operation] == _fully_connected) {
            memory_size += tinyANN->tensors[i].info[_input] * (height + tinyANN->tensors[i].info[_padding] * 2) *
                           (width + tinyANN->tensors[i].info[_padding] * 2);

            height = width = 1;

            continue;
        }

        memory_size += tinyANN->tensors[i].info[_input] * (height + tinyANN->tensors[i].info[_padding] * 2) *
                       (width + tinyANN->tensors[i].info[_padding] * 2);
        height = 1 + (height + 2 * tinyANN->tensors[i].info[_padding] - tinyANN->tensors[i].info[_kernel_size]) /
                         tinyANN->tensors[i].info[_stride];
        width = 1 + (width + 2 * tinyANN->tensors[i].info[_padding] - tinyANN->tensors[i].info[_kernel_size]) /
                        tinyANN->tensors[i].info[_stride];
    }

    // Memory for paramters
    for (int i = 0; i < tinyANN->total_layers - 1; i++) {
        if (tinyANN->tensors[i].info[_operation] == _convolution || tinyANN->tensors[i].info[_operation] == _fully_connected) {
            memory_size += tinyANN->tensors[i].info[_input] * tinyANN->tensors[i].info[_output] *
                           tinyANN->tensors[i].info[_kernel_size] * tinyANN->tensors[i].info[_kernel_size];
            memory_size += tinyANN->tensors[i].info[_output];
        }
    }

    tinyANN->memory_block.memory_start = (float*)malloc(memory_size * sizeof(float));
    if (!tinyANN->memory_block.memory_start) {
        fprintf(stderr, "ERROR TINY_ANN: Allocation error in allocateMemoryRegion()\n");
        return MEMORY_ALLOCATION_FAILED;
    }
    tinyANN->memory_block.size = memory_size;
    tinyANN->memory_block.memory_used = tinyANN->memory_block.memory_start;

    return SUCCESS;
}

int deallocateMemoryRegion(TinyANN* tinyANN) {
    if (tinyANN->memory_block.memory_start)
        free(tinyANN->memory_block.memory_start);
    return SUCCESS;
}
