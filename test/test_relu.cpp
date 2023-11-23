#include "layer/concrete/relu.hpp"
#include <catch2/catch_test_macros.hpp>
#include "cmath"

SCENARIO("test layer relu", "[layer_relu]") {

    using namespace nova_infer;

    GIVEN("one batch as input and one batch as output") {
        set_multi_sink();

        auto input = std::make_shared<Batchf>();
        std::vector<int> t1_shape = {1, 2, 2};
        std::vector<float> t1_data = {1.f, 2.f, 3.f, 4.f};
        Tensor<float> t1(t1_shape, t1_data);
        input->emplace_back(std::move(t1));


        auto output = std::make_shared<Batchf>();
        std::vector<int> t2_shape = {1, 2, 2};
        Tensor<float> t2(t2_shape);
        output->emplace_back(std::move(t2));


        std::string name = "relu_test";

        std::vector<std::string> input_name = {"example_in"};
        std::vector<std::string> output_name = {"example_out"};
        LayerReLU layer(name, input_name, output_name);

        layer.AssignInput(input);
        layer.AssignOutput(output);

        WHEN("execute relu") {
            layer.Forward();

            THEN("the output should be filled properly") {

                float val00 = std::max(input->at(0).ReadMatrix(0)(0,0), 0.f);
                REQUIRE(output->at(0).ReadMatrix(0)(0,0) == val00);

                float val10 = std::max(input->at(0).ReadMatrix(0)(1,0), 0.f);
                REQUIRE(output->at(0).ReadMatrix(0)(1,0) == val10);

                float val01 = std::max(input->at(0).ReadMatrix(0)(0,1), 0.f);
                REQUIRE(output->at(0).ReadMatrix(0)(0,1) == val01);

                float val11 = std::max(input->at(0).ReadMatrix(0)(1,1), 0.f);
                REQUIRE(output->at(0).ReadMatrix(0)(1,1) == val11);
            }
        }
    }
}