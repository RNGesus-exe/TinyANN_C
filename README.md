# TinyANN_C
A small lightweight artificial neural network library made in C to be used for only forward propogation by using pretrained model weights &amp; biases.

# How to use Library
The following steps are explained in their own respective section
```
1) Define network configuration
2) Add dataset
3) Add classes
4) Add parameters (weights & bias)
```
After the above steps, you need to:
```
mkdir build
cd build
cmake ..
make
./tinyann_cpp
```

## 1) Define the Network Configuration
Below is an example configuration of a neural network  
```
3 128 128
11
1 2 1 3 1 3 16
2 2 0 2 0 16 16
1 2 1 3 1 16 32
2 2 0 2 0 32 32
1 1 1 3 1 32 64
2 2 0 2 0 64 64
3 1 0 0 0 64 1024
4 1 0 1 1 1024 100
4 1 0 1 1 100 50
4 1 0 1 0 50 10
5 1 0 1 1 10 1
``` 
First, we need to define our input layer/image dimensions ```3 128 128``` (3 Channels, 128px Heights, 128px Width).  
Secondly, we will define the number of hidden layers + 1, We have '10' hidden layers above but we add '1' to account for the output layer.  
Lastly we will define each layer one by one with the following format:  
```
operation stride padding kernel_size activation in_dim out_dim <FORMAT>
Operation : Convolution(1) Max_Pool(2) Flatten(3) Fully_Connected(4) Output_Layer(5)
Stride : By default keep it 1 in each layer but you can increase it for certain layers such as convolution and max_pool
Padding : By default keep it 0 in each layer but you can increase it for certain layers such as convolution and max_pool
Kernel_Size : In flatten layer set this as 0 but in a fully connected layer set this as 1, You can change it accordingly in other layers
Input_Dimension : Input Dimension of Layer
Output_Dimension : Output Dimension of Layer
```
Note: Always flatten when going from convolution to fully connected layer, Make sure the last layer has 1 as the output dimension

## 2) Include the Dataset

Add your dataset in the ```extern/test_data``` directory.

## 3) Define Classes

Add your classes in the ```extern/classes.txt``` file, but make sure classes have the same case and spelling as dataset folders.

## 4) Add Parameters

Include the file from which weights and biases will be loaded for the neural network.
