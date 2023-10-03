#include "layer/concrete/linear.hpp"
#include <catch2/catch_test_macros.hpp>



SCENARIO("test layer linear", "[layer_linear]") {

    using namespace sky_infer;

    GIVEN("one batch as input and one batch as output") {
        set_multi_sink();

        auto input = std::make_shared<Batchf>();
        std::vector<int> t1_shape = {1, 2, 4};
        std::vector<float> t1_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        Tensor<float> t1(t1_shape, t1_data);
        input->emplace_back(std::move(t1));


        auto output = std::make_shared<Batchf>();
        std::vector<int> t2_shape = {1, 2, 2};
        Tensor<float> t2(t2_shape);
        output->emplace_back(std::move(t2));


        std::string name = "linear_test";

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights {
                {0,0,0,0},
                {1,1,1,1},
        };

        Eigen::RowVectorXf bias{{-10, 10}};

        LayerLinear layer(name, input, output, weights, true, bias);

        WHEN("execute linear") {
            layer.Forward();

            THEN("the output should be filled properly") {

                REQUIRE(output->at(0).ReadMatrix(0)(0,0) == -10.f);

                REQUIRE(output->at(0).ReadMatrix(0)(1,0) == -10.f);

                REQUIRE(output->at(0).ReadMatrix(0)(0,1) == 20.f);

                REQUIRE(output->at(0).ReadMatrix(0)(1,1) == 36.f);

            }
        }
    }
}