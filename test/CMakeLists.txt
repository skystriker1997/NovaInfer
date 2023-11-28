add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE Eigen3::Eigen util Catch2::Catch2WithMain)


add_executable(sigmoid_test sigmoid_test.cpp)
target_link_libraries(sigmoid_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(relu_test relu_test.cpp)
target_link_libraries(relu_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(flatten_test flatten_test.cpp)
target_link_libraries(flatten_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(linear_test linear_test.cpp)
target_link_libraries(linear_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(conv_test convolution_test.cpp)
target_link_libraries(conv_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(transposed_conv_test transposed_convolution_test.cpp)
target_link_libraries(transposed_conv_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(softmax_test softmax_test.cpp)
target_link_libraries(softmax_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(cat_test cat_test.cpp)
target_link_libraries(cat_test PUBLIC layer Catch2::Catch2WithMain )


add_executable(openmp_test openmp_test.cpp)
target_link_libraries(openmp_test PUBLIC OpenMP::OpenMP_CXX )


