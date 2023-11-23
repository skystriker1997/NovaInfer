#include "layer/concrete/softmax.hpp"
#include <catch2/catch_test_macros.hpp>



SCENARIO("test layer softmax", "[layer_softmax]") {

    using namespace nova_infer;

    GIVEN("one batch as input and one batch as output") {
        set_multi_sink();

        auto input = std::make_shared<Batchf>();
        std::vector<int> t1_shape = {1, 2, 2};
        std::vector<float> t1_data = {1.f, 1.f, 1.f, 1.f};
        Tensor<float> t1(t1_shape, t1_data);
        input->emplace_back(std::move(t1));


        auto output = std::make_shared<Batchf>();
        std::vector<int> t2_shape = {1, 2, 2};
        Tensor<float> t2(t2_shape);
        output->emplace_back(std::move(t2));


        std::string name = "softmax_test";

        int target_dim = 1;

        std::vector<std::string> input_name = {"example_in"};
        std::vector<std::string> output_name = {"example_out"};

        LayerSoftmax layer(name, input_name, output_name, target_dim);

        layer.AssignInput(input);
        layer.AssignOutput(output);

        WHEN("execute softmax") {
            layer.Forward();

            THEN("the output should be filled properly") {

                REQUIRE(output->at(0).ReadMatrix(0)(0,0) == 0.5);

                REQUIRE(output->at(0).ReadMatrix(0)(0,1) == 0.5);

                REQUIRE(output->at(0).ReadMatrix(0)(1,0) == 0.5);

                REQUIRE(output->at(0).ReadMatrix(0)(1,1) == 0.5);

            }
        }
    }
}

