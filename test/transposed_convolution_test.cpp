#include "layer/concrete/transposed_convolution.hpp"
#include <catch2/catch_test_macros.hpp>



SCENARIO("test layer transposed convolution", "[layer_transposed_convolution]") {

    using namespace nova_infer;

    GIVEN("one batch as input and one batch as output") {
        set_multi_sink();

        auto input = std::make_shared<Batchf>();
        std::vector<int> t1_shape = {1, 2, 2};
        std::vector<float> t1_data = {1,2,3,4};
        Tensor<float> t1(t1_shape, t1_data);
        input->emplace_back(std::move(t1));

        auto output = std::make_shared<Batchf>();
        std::vector<int> t2_shape = {2, 2, 2};
        Tensor<float> t2(t2_shape);
        output->emplace_back(std::move(t2));

        Batchf weights;
        std::vector<int> kernel_shape = {1, 3, 3};
        std::vector<float> kernel_data = {1,0,1,0,0,0,1,0,1};
        Tensor<float> kernel(kernel_shape, kernel_data);
        weights = {kernel, kernel};

        int padding_h = 1;
        int padding_w = 1;
        int stride_h = 1;
        int stride_w = 1;
        int groups = 1;
        int output_padding_h = 0;
        int output_padding_w = 0;
        Eigen::RowVectorXf bias(2);
        bias << 1.f,-1.f;


        std::string name = "transposed_convolution_test";


        std::vector<std::string> input_name = {"example_in"};
        std::vector<std::string> output_name = {"example_out"};

        LayerTransposedConvolution layer(name, input_name, output_name, weights, true, bias, padding_h, padding_w, stride_h, stride_w, groups, output_padding_h, output_padding_w);

        layer.AttachInput(input);
        layer.AttachOutput(output);

        WHEN("execute transposed convolution") {
            layer.Forward();

            THEN("the output should be filled properly") {

                REQUIRE(output->at(0).ReadMatrix(0)(0,0) == 5.f);
                REQUIRE(output->at(0).ReadMatrix(1)(0,1) == 2.f);

            }
        }
    }
}