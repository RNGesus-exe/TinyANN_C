#include "../include/cnn.h"

#include <dirent.h>
#include <opencv4/opencv2/opencv.hpp>

void loadDataset(const char* dataset_path, cv::Mat& image) {
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
            loadDataset(subfolderPath, image);
        } else if (entry->d_type == DT_REG) { // Check if it's a regular file
            // Get the file name
            const char* fileName = entry->d_name;

            // Check if the file is an image (you can add more image formats if needed)
            const char* extension = strrchr(fileName, '.');
            if (extension != NULL && (strcasecmp(extension, ".jpg") == 0 || strcasecmp(extension, ".jpeg") == 0)) {

                // Construct the full image path
                char imagePath[256];
                snprintf(imagePath, sizeof(imagePath), "%s/%s", dataset_path, fileName);
                std::cout << imagePath << std::endl;

                // Read the image
                image = cv::imread(imagePath, cv::IMREAD_COLOR);

                if (image.empty()) {
                    std::cout << "Error: Could not read the image.\n";
                    return;
                }

                cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

                // Perform Inference in this layer

                // inference();
            }
        }
    }

    // Close the directory
    closedir(directory);
}

int main() {

    const char* network_config_path = "../extern/network_config.txt";
    // const char* images_path = "../extern/Emperor Tamarin";
    const char* param_path = "../extern/parameters.txt";

    TinyANN tinyANN;
    initNetwork(&tinyANN, network_config_path, param_path);
    destroyNetwork(&tinyANN);
    // cv::Mat image;

    // loadDataset(images_path, image);

    // printf("Total Images = %d\n", test_set_size);
    // printf("Accuracy = %f\n", (float)true_positives / test_set_size * 100);

    return 0;
}
