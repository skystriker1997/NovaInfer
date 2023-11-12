cmake_minimum_required(VERSION 3.23)
project(sky_infer)

list(APPEND CMAKE_MODULE_PATH /usr/local/lib)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Catch2 3 REQUIRED)
find_package(OpenMP REQUIRED)

add_library(util source/inf_util.cpp)
target_include_directories(util PUBLIC "./include")
target_link_libraries(util PUBLIC spdlog::spdlog)



aux_source_directory( source/layer/concrete CONCRETE_LAYERS)
add_library(layer ${CONCRETE_LAYERS})
target_include_directories(layer PUBLIC "./include")
target_link_libraries(layer PUBLIC Eigen3::Eigen util OpenMP::OpenMP_CXX)


add_library(graph source/graph.cpp)
target_include_directories(graph PUBLIC "./include")
target_link_libraries(graph PUBLIC layer)



########

add_executable(tensor_test test/test_tensor.cpp)
target_include_directories(tensor_test PUBLIC "./include")
target_link_libraries(tensor_test PRIVATE Eigen3::Eigen util Catch2::Catch2WithMain)



add_executable(sigmoid_test test/test_sigmoid.cpp)
target_include_directories(sigmoid_test PUBLIC "./include")
target_link_libraries(sigmoid_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(relu_test test/test_relu.cpp)
target_include_directories(relu_test PUBLIC "./include")
target_link_libraries(relu_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(flatten_test test/test_flatten.cpp)
target_include_directories(flatten_test PUBLIC "./include")
target_link_libraries(flatten_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(linear_test test/test_linear.cpp)
target_include_directories(linear_test PUBLIC "./include")
target_link_libraries(linear_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(conv_test test/test_convolution.cpp)
target_include_directories(conv_test PUBLIC "./include")
target_link_libraries(conv_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(transposed_conv_test test/test_transposed_convolution.cpp)
target_include_directories(transposed_conv_test PUBLIC "./include")
target_link_libraries(transposed_conv_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(softmax_test test/test_softmax.cpp)
target_include_directories(softmax_test PUBLIC "./include")
target_link_libraries(softmax_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(openmp_test test/test_openmp.cpp)
target_include_directories(openmp_test PUBLIC "./include")
target_link_libraries(openmp_test PUBLIC OpenMP::OpenMP_CXX )
