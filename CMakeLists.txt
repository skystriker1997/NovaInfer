cmake_minimum_required(VERSION 3.20.53)
project(NovaInfer)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Catch2 3 REQUIRED)
find_package(OpenMP REQUIRED)
#find_package(MKL CONFIG PATHS $ENV{MKLROOT})
find_package(MKL CONFIG)
find_package(IntelSYCL)

if(MKL_FOUND AND IntelSYCL_FOUND)
    message(STATUS "Intel MKL enabled")
# else()
#     message(STATUS "Intel MKL not found")
endif()

include_directories(include/)
link_libraries(OpenMP::OpenMP_CXX)

add_library(util source/inf_util.cpp)
target_link_libraries(util PUBLIC spdlog::spdlog)

aux_source_directory(source/pnnx PNNX_PARTS)
add_library(pnnx ${PNNX_PARTS})

aux_source_directory(source/layer/concrete CONCRETE_LAYERS)
add_library(layer ${CONCRETE_LAYERS})
target_link_libraries(layer PUBLIC Eigen3::Eigen util pnnx)

if(MKL_FOUND AND IntelSYCL_FOUND)
    target_compile_options(layer PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(layer PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(layer PUBLIC $<LINK_ONLY:MKL::MKL>)
endif()


add_library(graph source/graph.cpp)
target_link_libraries(graph PUBLIC layer)


add_subdirectory(demo)

add_subdirectory(test)




