#include "layer/concrete/convolution.hpp"
#include <catch2/catch_test_macros.hpp>



SCENARIO("test layer convolution", "[layer_convolution]") {

    using namespace sky_infer;

    GIVEN("one batch as input and one batch as output") {
        set_multi_sink();

        auto input = std::make_shared<Batchf>();
        std::vector<int> t1_shape = {9, 4, 4};
        std::vector<float> t1_data(144, 1.f);
        Tensor<float> t1(t1_shape, t1_data);
        input->emplace_back(std::move(t1));

        auto output = std::make_shared<Batchf>();
        std::vector<int> t2_shape = {3, 3, 3};
        Tensor<float> t2(t2_shape);
        output->emplace_back(std::move(t2));

        Batchf weights;
        std::vector<int> kernel_shape = {3, 2, 2};
        std::vector<float> kernel_data(12, 1.f);
        Tensor<float> kernel(kernel_shape, kernel_data);
        weights = {kernel, kernel, kernel};

        int padding_h = 1;
        int padding_w = 1;
        int stride_h = 2;
        int stride_w = 2;
        int groups = 3;
        Eigen::RowVectorXf bias(3);
        bias << 1.f,1.f,1.f;


        std::string name = "convolution_test";


        std::vector<std::string> input_name = {"example_in"};
        std::vector<std::string> output_name = {"example_out"};

        LayerConvolution layer(name, input_name, output_name, weights, true, bias, padding_h, padding_w, stride_h, stride_w, groups);

        layer.AssignInput(input);
        layer.AssignOutput(output);

        WHEN("execute convolution") {
            layer.Forward();

            THEN("the output should be filled properly") {

                REQUIRE(output->at(0).ReadMatrix(0)(0,0) == 4.f);
                REQUIRE(output->at(0).ReadMatrix(0)(0,1) == 7.f);

            }
        }
    }
}