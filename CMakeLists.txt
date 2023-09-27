cmake_minimum_required(VERSION 3.23)
project(sky_infer)

list(APPEND CMAKE_MODULE_PATH /usr/local/lib)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Catch2 3 REQUIRED)


aux_source_directory( source/layer/concrete CONCRETE_LAYERS)
add_library(layer ${CONCRETE_LAYERS})
target_include_directories(layer PUBLIC "./include")
target_link_libraries(layer PUBLIC Eigen3::Eigen spdlog::spdlog)


add_library(graph source/graph.cpp)
target_include_directories(graph PUBLIC "./include")
target_link_libraries(graph PUBLIC layer)


add_executable(tensor_test test/test_tensor.cpp)
target_include_directories(tensor_test PUBLIC "./include")
target_link_libraries(tensor_test PRIVATE Eigen3::Eigen spdlog::spdlog Catch2::Catch2WithMain)











