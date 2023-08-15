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
            tinyANN->tensors[i].weight_start = tinyANN->tensors[i].bias_start = tinyANN->tensors[i].bias_end = tinyANN->tensors[i].weight_end = NULL;
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
        for (int out = 0; out < tinyANN->tensors[t].info[_output]; out++) {
            fscanf(param_config, "%f", &tinyANN->tensors[t].bias_start[out]);
        }
    }

    fclose(param_config);

    return SUCCESS;
}

int inference(TinyANN* tinyANN, float* image) {

    size_t padding = tinyANN->tensors[0].info[_padding];
    size_t pad_height = tinyANN->tensors[0].height + 2 * padding;
    size_t pad_width = tinyANN->tensors[0].width + 2 * padding;

    for (int f = 0; f < tinyANN->image_filters; f++) {
        for (int i = 0; i < tinyANN->image_rows; i++) {
            for (int j = 0; j < tinyANN->image_cols; j++) {
                tinyANN->tensors[0].start[f * pad_height * pad_width + ((i + padding) * pad_width + (j + padding))] =
                    image[f * tinyANN->image_rows * tinyANN->image_cols + (i * tinyANN->image_cols + j)];
            }
        }
    }

    for (int l = 0; l < tinyANN->total_layers - 1; l++) {
        if (tinyANN->tensors[l].info[_operation] == _convolution) {
            convolution(tinyANN, l);
            if (tinyANN->tensors[l].info[_activation] == _relu) {
                relu(tinyANN, l + 1);
            }
        } else if (tinyANN->tensors[l].info[_operation] == _maxpool) {
            max_pool(tinyANN, l);
        } else if (tinyANN->tensors[l].info[_operation] == _flatten) {
            flatten(tinyANN, l);
        } else if (tinyANN->tensors[l].info[_operation] == _fully_connected) {
            fully_connected(tinyANN, l);
            if (tinyANN->tensors[l].info[_activation] == _relu) {
                relu(tinyANN, l + 1);
            }
        }
    }

    int max_ind = 0;
    for (int i = 1; i < tinyANN->tensors[tinyANN->total_layers - 1].channels; i++) {
        if (tinyANN->tensors[tinyANN->total_layers - 1].start[max_ind] < tinyANN->tensors[tinyANN->total_layers - 1].start[i]) {
            max_ind = i;
        }
    }

#if 0
    FILE* file = fopen("../extern/out_feature_map.txt", "wb");

    for (int l = 0; l < tinyANN->total_layers; l++) {
        padding = tinyANN->tensors[l].info[_padding];
        pad_height = tinyANN->tensors[l].height + 2 * padding;
        pad_width = tinyANN->tensors[l].width + 2 * padding;

        fprintf(file, "\n==========Layer %d: %ld %ld %ld %ld============\n", l + 1, tinyANN->tensors[l].channels, pad_height, pad_width,
                tinyANN->tensors[l].info[_padding]);

        for (int f = 0; f < tinyANN->tensors[l].channels; f++) {
            for (int i = 0; i < pad_height; i++) {
                for (int j = 0; j < pad_width; j++) {
                    fprintf(file, "%f ", tinyANN->tensors[l].start[f * pad_height * pad_width + (i * pad_width + j)]);
                }
                fprintf(file, "\n");
            }
            fprintf(file, "\n\n");
        }
        fprintf(file, "\n\n\n");
    }
    fclose(fikle);
#endif

    return max_ind;
}

void fully_connected(TinyANN* tinyANN, size_t layer_no) {
    for (int n = 0; n < tinyANN->tensors[layer_no].info[_output]; n++) {
        tinyANN->tensors[layer_no + 1].start[n] = 0;
        for (int m = 0; m < tinyANN->tensors[layer_no].info[_input]; m++) {
            tinyANN->tensors[layer_no + 1].start[n] +=
                tinyANN->tensors[layer_no].start[m] * tinyANN->tensors[layer_no].weight_start[n * tinyANN->tensors[layer_no].info[_input] + m];
        }
        tinyANN->tensors[layer_no + 1].start[n] += tinyANN->tensors[layer_no].bias_start[n];
    }
}

void flatten(TinyANN* tinyANN, size_t layer_no) {
    int i = 0;
    for (float* ptr = tinyANN->tensors[layer_no].start; ptr != tinyANN->tensors[layer_no].end; ptr++) {
        tinyANN->tensors[layer_no + 1].start[i++] = *ptr;
    }
}

void relu(TinyANN* tinyANN, size_t layer_no) {

    size_t padding = tinyANN->tensors[layer_no].info[_padding];
    size_t pad_height = tinyANN->tensors[layer_no].height + 2 * padding;
    size_t pad_width = tinyANN->tensors[layer_no].width + 2 * padding;

    for (int c = 0; c < tinyANN->tensors[layer_no].channels; c++) {
        for (int i = 0; i < tinyANN->tensors[layer_no].height; i++) {
            for (int j = 0; j < tinyANN->tensors[layer_no].width; j++) {
                if (tinyANN->tensors[layer_no].start[c * pad_height * pad_width + ((i + padding) * pad_width + (j + padding))] < 0) {
                    tinyANN->tensors[layer_no].start[c * pad_height * pad_width + ((i + padding) * pad_width + (j + padding))] = 0.0f;
                }
            }
        }
    }
}

void max_pool(TinyANN* tinyANN, size_t layer_no) {
    size_t new_padding = tinyANN->tensors[layer_no + 1].info[_padding];
    size_t new_pad_height = tinyANN->tensors[layer_no + 1].height + 2 * new_padding;
    size_t new_pad_width = tinyANN->tensors[layer_no + 1].width + 2 * new_padding;

    size_t padding = tinyANN->tensors[layer_no].info[_padding];
    size_t pad_height = tinyANN->tensors[layer_no].height + 2 * padding;
    size_t pad_width = tinyANN->tensors[layer_no].width + 2 * padding;
    size_t kernel_size = tinyANN->tensors[layer_no].info[_kernel_size];
    size_t stride = tinyANN->tensors[layer_no].info[_stride];
    size_t in_filters = tinyANN->tensors[layer_no].info[_input];
    size_t out_filters = tinyANN->tensors[layer_no].info[_output];

    size_t stride_x, stride_y;

    for (int out_f = 0; out_f < out_filters; out_f++) {
        for (int row = 0; row < tinyANN->tensors[layer_no + 1].height; row++) {
            for (int col = 0; col < tinyANN->tensors[layer_no + 1].width; col++) {

                float max_val = INT32_MIN;
                stride_x = stride * row;
                stride_y = stride * col;

                for (int x = 0; x < kernel_size; x++) {
                    for (int y = 0; y < kernel_size; y++) {
                        if (tinyANN->tensors[layer_no].start[out_f * pad_height * pad_width + ((x + stride_x) * pad_width + (stride_y + y))] >
                            max_val) {
                            max_val =
                                tinyANN->tensors[layer_no].start[out_f * pad_height * pad_width + ((x + stride_x) * pad_width + (stride_y + y))];
                        }
                    }
                }

                tinyANN->tensors[layer_no + 1]
                    .start[out_f * new_pad_height * new_pad_width + ((row + new_padding) * new_pad_width + (col + new_padding))] = max_val;
            }
        }
    }

    // Clean up the previous feature map
    for (float* ptr = tinyANN->tensors[layer_no].start; ptr != tinyANN->tensors[layer_no].end; ptr++) {
        *ptr = 0.0f;
    }
}

void convolution(TinyANN* tinyANN, size_t layer_no) {

    size_t new_padding = tinyANN->tensors[layer_no + 1].info[_padding];
    size_t new_pad_height = tinyANN->tensors[layer_no + 1].height + 2 * new_padding;
    size_t new_pad_width = tinyANN->tensors[layer_no + 1].width + 2 * new_padding;

    size_t padding = tinyANN->tensors[layer_no].info[_padding];
    size_t pad_height = tinyANN->tensors[layer_no].height + 2 * padding;
    size_t pad_width = tinyANN->tensors[layer_no].width + 2 * padding;
    size_t kernel_size = tinyANN->tensors[layer_no].info[_kernel_size];
    size_t stride = tinyANN->tensors[layer_no].info[_stride];
    size_t in_filters = tinyANN->tensors[layer_no].info[_input];
    size_t out_filters = tinyANN->tensors[layer_no].info[_output];

    size_t stride_x, stride_y;

    for (int out_f = 0; out_f < out_filters; out_f++) {
        for (int row = 0; row < tinyANN->tensors[layer_no + 1].height; row++) {
            for (int col = 0; col < tinyANN->tensors[layer_no + 1].width; col++) {

                float dot_product = 0.0f;
                stride_x = stride * row;
                stride_y = stride * col;

                for (int in_f = 0; in_f < in_filters; in_f++) {

                    for (int x = 0; x < kernel_size; x++) {
                        for (int y = 0; y < kernel_size; y++) {

                            dot_product +=
                                tinyANN->tensors[layer_no].start[in_f * pad_height * pad_width + ((x + stride_x) * pad_width + (stride_y + y))] *
                                tinyANN->tensors[layer_no].weight_start[out_f * in_filters * kernel_size * kernel_size +
                                                                        (in_f * kernel_size * kernel_size + (x * kernel_size + y))];
                        }
                    }
                }
                tinyANN->tensors[layer_no + 1]
                    .start[out_f * new_pad_height * new_pad_width + ((row + new_padding) * new_pad_width + (col + new_padding))] =
                    dot_product + tinyANN->tensors[layer_no].bias_start[out_f];
            }
        }
    }

    // Clean up the previous feature map
    for (float* ptr = tinyANN->tensors[layer_no].start; ptr != tinyANN->tensors[layer_no].end; ptr++) {
        *ptr = 0.0f;
    }
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

        memory_size +=
            tinyANN->tensors[i].info[_input] * (height + tinyANN->tensors[i].info[_padding] * 2) * (width + tinyANN->tensors[i].info[_padding] * 2);
        height = 1 + (height + 2 * tinyANN->tensors[i].info[_padding] - tinyANN->tensors[i].info[_kernel_size]) / tinyANN->tensors[i].info[_stride];
        width = 1 + (width + 2 * tinyANN->tensors[i].info[_padding] - tinyANN->tensors[i].info[_kernel_size]) / tinyANN->tensors[i].info[_stride];
    }

    // Memory for paramters
    for (int i = 0; i < tinyANN->total_layers - 1; i++) {
        if (tinyANN->tensors[i].info[_operation] == _convolution || tinyANN->tensors[i].info[_operation] == _fully_connected) {
            memory_size += tinyANN->tensors[i].info[_input] * tinyANN->tensors[i].info[_output] * tinyANN->tensors[i].info[_kernel_size] *
                           tinyANN->tensors[i].info[_kernel_size];
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
