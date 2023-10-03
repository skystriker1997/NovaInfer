#include "layer/concrete/flatten.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>


SCENARIO("test layer flatten", "[layer_flatten]") {

    using namespace sky_infer;

    GIVEN("one batch as input and one batch as output") {
        set_multi_sink();

        auto input = std::make_shared<Batchf>();
        std::vector<int> t1_shape = {1, 2, 2};
        std::vector<float> t1_data = {1.f, 2.f, 3.f, 4.f};
        Tensor<float> t1(t1_shape, t1_data);
        input->emplace_back(std::move(t1));


        auto output = std::make_shared<Batchf>();
        std::vector<int> t2_shape = {1, 1, 4};
        Tensor<float> t2(t2_shape);
        output->emplace_back(std::move(t2));


        std::string name = "flatten_test";

        int flatten_start_dim = 1;
        int flatten_end_dim = 3;

        LayerFlatten layer(name, input, output, flatten_start_dim, flatten_end_dim);

        WHEN("execute flatten") {
            layer.Forward();

            THEN("the output should be filled properly") {

                REQUIRE(output->at(0).ReadMatrix(0)(0,0) == 1.f);

                REQUIRE(output->at(0).ReadMatrix(0)(0,1) == 2.f);

                REQUIRE(output->at(0).ReadMatrix(0)(0,2) == 3.f);

                REQUIRE(output->at(0).ReadMatrix(0)(0,3) == 4.f);

            }
        }
    }
}