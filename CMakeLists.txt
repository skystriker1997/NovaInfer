cmake_minimum_required(VERSION 3.23)
project(sky_infer)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)


add_library(attr source/attribute.cpp)
target_include_directories(attr PUBLIC "./include")
target_link_libraries(attr PUBLIC glog::glog Eigen3::Eigen)


add_library(opd source/operand.cpp)
target_include_directories(opd PUBLIC "./include")
target_link_libraries(opd PUBLIC glog::glog Eigen3::Eigen)


add_library(opt source/operator.cpp)
target_include_directories(opt PUBLIC "./include")
target_link_libraries(opt PUBLIC attr opd)


add_library(parser source/layer/concrete/expression/parser.cpp)
target_include_directories(parser PUBLIC "./include")
target_link_libraries(parser PUBLIC opd opt)


aux_source_directory( source/layer/concrete CONCRETE_LAYERS)
add_library(layer ${CONCRETE_LAYERS} source/layer/concrete/expression/expression.cpp)
target_include_directories(layer PUBLIC "./include")
target_link_libraries(layer PUBLIC opd opt parser)


add_library(graph source/graph.cpp)
target_include_directories(graph PUBLIC "./include")
target_link_libraries(graph PUBLIC layer)


add_executable(tensor_test test/test_tensor.cpp)
target_include_directories(tensor_test PUBLIC "./include")
target_link_libraries(tensor_test PUBLIC glog::glog Eigen3::Eigen)








