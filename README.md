# NovaInfer



## Overview

Written by C++, Novainfer is a deep learning inference framework in development. I name it by nova becasue I have been holding a high hope to improve it step by step as I am making progress in the area of deep learning. My aim is to make it be able to deploy most models and support CUDA acceleration.

## Development environment and libraries used

* C++ standard : C++17
* model format : [PNNX](https://github.com/Tencent/ncnn/tree/master/tools/pnnx)
* linear algebra library : [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)
* parallelisation library : OpenMP
* log library : [spdlog](https://github.com/gabime/spdlog)
* unit test library : [Catch2](https://github.com/catchorg/Catch2)

## Demo

### Unet

🥳 **_update_**

I am working to make NovaInfer perfectly support unet. The test case is displayed below, so I am trying to solve the issue under the hood 🧐

 <img src="./images/unet_car_input.png" style="zoom:67%;" /> 

<img src="./images/unet_car_output.jpg" style="zoom:67%;" />





