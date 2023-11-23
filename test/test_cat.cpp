#include "layer/concrete/cat.hpp"
#include <catch2/catch_test_macros.hpp>



SCENARIO("test layer cat", "[layer_cat]") {

    using namespace nova_infer;

    GIVEN("two batches as inputs and one batch as output") {
        set_multi_sink();

        auto input1 = std::make_shared<Batchf>();
        std::vector<int> t1_shape = {8, 4, 4};
        std::vector<float> t1_data(128, 1.f);
        Tensor<float> t1(t1_shape, t1_data);
        input1->emplace_back(std::move(t1));


        auto input2 = std::make_shared<Batchf>();
        std::vector<int> t2_shape = {3, 4, 4};
        std::vector<float> t2_data(48, 1.f);
        Tensor<float> t2(t2_shape, t2_data);
        input2->emplace_back(std::move(t2));


        auto output = std::make_shared<Batchf>();
        std::vector<int> t3_shape = {11, 4, 4};
        Tensor<float> t3(t3_shape);
        output->emplace_back(std::move(t3));

        int dim = 1;

        std::string name = "cat_test";

        std::vector<std::string> input_name = {"example_in1", "example_in2"};
        std::vector<std::string> output_name = {"example_out"};

        LayerCat layer(name, input_name, output_name, dim);

        layer.AssignInput({input1, input2});
        layer.AssignOutput(output);

        WHEN("execute cat") {
            layer.Forward();

            THEN("the output should be filled properly") {

            }
        }
    }
}