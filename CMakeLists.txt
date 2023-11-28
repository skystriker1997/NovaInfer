cmake_minimum_required(VERSION 3.23)
project(NovaInfer)

list(APPEND CMAKE_MODULE_PATH /usr/local/lib)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Catch2 3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(include)


add_library(util source/inf_util.cpp)
target_link_libraries(util PUBLIC spdlog::spdlog)


aux_source_directory(source/layer/concrete CONCRETE_LAYERS)
add_library(layer ${CONCRETE_LAYERS})
target_link_libraries(layer PUBLIC Eigen3::Eigen util OpenMP::OpenMP_CXX)


add_library(graph source/graph.cpp)
target_link_libraries(graph PUBLIC layer)


add_subdirectory(demo)

add_subdirectory(test)


