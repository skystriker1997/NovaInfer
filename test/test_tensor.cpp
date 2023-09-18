#include "tensor/tensor.hpp"


int main(int argc, char* argv[]) {
    using namespace sky_infer;
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "../log";
    FLAGS_alsologtostderr = true;

    std::vector<float> data1 = {1,1,1,1};
    std::vector<float> data2 = {1,2,3,4};

    Tensor<float> x(std::vector<int>{1,2,2}, data1, true);
    Tensor<float> y(std::vector<int>{1,2,2}, data2, true);

    std::cout << "Tensor x: " << std::endl;
    x.Print();

    std::cout << "Tensor y: " << std::endl;
    y.Print();

    std::cout << "Tensor x*y: " << std::endl;
    Tensor<float>{x*y}.Print();

    std::cout << "Tensor x+y: " << std::endl;
    Tensor<float>{x+y}.Print();

    std::cout << "Tensor x%y: " << std::endl;
    Tensor<float>{x%y}.Print();

    x.Padding(std::vector<int>{1,1,1,1}, 0);
    std::cout << "Tensor x after padding 0: " << std::endl;
    x.Print();


    return 0;
};
