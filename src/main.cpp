#include "../include/cnn.h"

#include <dirent.h>
#include <opencv4/opencv2/opencv.hpp>

void loadDataset(TinyANN* tinyANN, const char* dataset_path, cv::Mat& image, float* flatten_image, int* test_set_size, int* true_positives,
                 std::vector<std::string>& classes) {
    DIR* directory;
    struct dirent* entry;

    // Open the directory
    directory = opendir(dataset_path);
    if (directory == NULL) {
        fprintf(stderr, "Failed to open directory: %s\n", dataset_path);
        return;
    }

    // Read the directory entries
    while ((entry = readdir(directory)) != NULL) {
        if (entry->d_type == DT_DIR) { // Check if it's a subdirectory
            // Ignore the current directory (.) and parent directory (..)
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }

            // Construct the subfolder path
            char subfolderPath[256];
            strncpy(subfolderPath, dataset_path, sizeof(subfolderPath));
            strncat(subfolderPath, "/", sizeof(subfolderPath) - strlen(subfolderPath) - 1);
            strncat(subfolderPath, entry->d_name, sizeof(subfolderPath) - strlen(subfolderPath) - 1);

            // Read images recursively in the subfolder
            loadDataset(tinyANN, subfolderPath, image, flatten_image, test_set_size, true_positives, classes);
        } else if (entry->d_type == DT_REG) { // Check if it's a regular file
            // Get the file name
            const char* fileName = entry->d_name;

            // Check if the file is an image (you can add more image formats if needed)
            const char* extension = strrchr(fileName, '.');
            if (extension != NULL && (strcasecmp(extension, ".jpg") == 0 || strcasecmp(extension, ".jpeg") == 0)) {

                // Construct the full image path
                char imagePath[256];
                snprintf(imagePath, sizeof(imagePath), "%s/%s", dataset_path, fileName);
                // std::cout << imagePath << std::endl;

                // Read the image
                image = cv::imread(imagePath, cv::IMREAD_COLOR);

                if (image.empty()) {
                    std::cout << "Error: Could not read the image.\n";
                    return;
                }

                // Preprocessing
                cv::Size new_size(tinyANN->image_rows, tinyANN->image_cols);
                cv::resize(image, image, new_size);

                cv::Mat channels[tinyANN->image_filters];
                cv::split(image, channels);

                int ind = 0;
                for (int f = 0; f < tinyANN->image_filters; f++) {
                    for (int i = 0; i < tinyANN->image_rows; i++) {
                        for (int j = 0; j < tinyANN->image_cols; j++) {
                            flatten_image[ind++] = ((float)channels[tinyANN->image_filters - f - 1].at<uchar>(i, j) - (255 * 0.5f)) / (255 * 0.5f);
                        }
                    }
                }

                // Perform forward pass
                int res = inference(tinyANN, flatten_image);

                const char* class_name = strrchr(dataset_path, '/') + 1;
                if (strcmp(class_name, classes[res].c_str()) == 0) {
                    (*true_positives)++;
                }
                (*test_set_size)++;

                image.release();
            }
        }
    }

    // Close the directory
    closedir(directory);
}

void writeParamToFile(TinyANN* tinyANN, const char* path) {
    FILE* param_out = fopen(path, "wb");

    for (int t = 0; t < tinyANN->total_layers; t++) {
        if (tinyANN->tensors[t].info[_operation] != _convolution && tinyANN->tensors[t].info[_operation] != _fully_connected)
            continue;

        size_t out_filters = tinyANN->tensors[t].info[_output];
        size_t in_filters = tinyANN->tensors[t].info[_input];
        size_t kernel_size = tinyANN->tensors[t].info[_kernel_size];

        for (int out = 0; out < out_filters; out++) {
            for (int in = 0; in < in_filters; in++) {
                for (int r = 0; r < kernel_size; r++) {
                    for (int c = 0; c < kernel_size; c++) {
                        fprintf(param_out, "%f ",
                                tinyANN->tensors[t].weight_start[out * in_filters * kernel_size * kernel_size +
                                                                 (in * kernel_size * kernel_size + (r * kernel_size + c))]);
                    }
                    fprintf(param_out, "\n");
                }

                fprintf(param_out, "\n\n");
            }

            fprintf(param_out, "\n\n\n");
        }
        fprintf(param_out, "\n\n\n\n");
        for (int out = 0; out < tinyANN->tensors[t].info[_output]; out++) {
            fprintf(param_out, "%f ", tinyANN->tensors[t].bias_start[out]);
        }
        fprintf(param_out, "\n\n\n\n");
    }

    fclose(param_out);
}

std::vector<std::string> loadClasses(const int total_classes, const char* classes_path) {
    FILE* class_file = fopen(classes_path, "rb");

    std::vector<std::string> classes;

    if (class_file == NULL) {
        fprintf(stderr, "ERROR TINYANN: Classes could not be read\n");
        return classes;
    }

    char class_name[256];
    for (int i = 0; i < total_classes; i++) {
        if (fgets(class_name, sizeof(class_name), class_file) != NULL) {
            size_t length = strlen(class_name);
            if (length > 0 && class_name[length - 1] == '\n') {
                class_name[length - 1] = '\0';
            }
        }
        classes.push_back(class_name);
    }

    fclose(class_file);

    return classes;
}

int main() {

    const char* network_config_path = "../extern/network_config.txt";
    const char* images_path = "../extern/test_data";
    const char* param_path = "../extern/parameters.txt";
    const char* classes_path = "../extern/classes.txt";

    int test_set_size = 0;
    int true_positives = 0;
    TinyANN tinyANN;
    cv::Mat image;

    initNetwork(&tinyANN, network_config_path, param_path);
    float flatten_image[tinyANN.image_filters * tinyANN.image_rows * tinyANN.image_cols];

    std::vector<std::string> classes = loadClasses(tinyANN.tensors[tinyANN.total_layers - 1].channels, classes_path);
    loadDataset(&tinyANN, images_path, image, flatten_image, &test_set_size, &true_positives, classes);

    printf("Total Images = %d\n", test_set_size);
    printf("Accuracy = %f\n", (float)true_positives / test_set_size * 100);

    destroyNetwork(&tinyANN);
    return 0;
}
