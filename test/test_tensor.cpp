#include "tensor/tensor.hpp"
#include <catch2/catch_test_macros.hpp>





SCENARIO("tensor can be operated", "[tensor]") {

    using namespace sky_infer;

    GIVEN("two tensors x, y with some items") {
        set_multi_sink();
        std::vector<float> data1 = {1, 1, 1, 1};
        std::vector<float> data2 = {1, 2, 3, 4};

        Tensor<float> x(std::vector<int>{1, 2, 2}, data1, true);
        Tensor<float> y(std::vector<int>{1, 2, 2}, data2, true);

        REQUIRE(y.ReadMatrix(0)(1,0)==2);
        REQUIRE(y.ReadMatrix(0)(0,1)==3);
        REQUIRE(y.Max()==4);
        REQUIRE(y.Min()==1);

        WHEN("tensor z = x + y") {
            Tensor<float> z = x + y;

            THEN("z equals adding coefficients from x and y correspondingly") {
                REQUIRE(z.ReadShape() == x.ReadShape());
                REQUIRE(z.ReadMatrix(0)(0,0)==2);
                REQUIRE(z.ReadMatrix(0)(1,0)==3);
                REQUIRE(z.ReadMatrix(0)(0,1)==4);
                REQUIRE(z.ReadMatrix(0)(1,1)==5);
            }
        }
        WHEN("tensor z = x * y") {
            Tensor<float> z = x * y;

            THEN("z follows the rule of matrix dot product") {
                REQUIRE(z.ReadShape() == std::vector<int>{1,2,2});
                REQUIRE(z.ReadMatrix(0)(0,0)==3);
                REQUIRE(z.ReadMatrix(0)(1,0)==3);
                REQUIRE(z.ReadMatrix(0)(0,1)==7);
                REQUIRE(z.ReadMatrix(0)(1,1)==7);
            }
        }
        WHEN("tensor z = x % y") {
            Tensor<float> z = x % y;

            THEN("z equals multiplying coefficients from x and y correspondingly") {
                REQUIRE(z.ReadShape() == x.ReadShape());
                REQUIRE(z.ReadMatrix(0)(0,0)==1);
                REQUIRE(z.ReadMatrix(0)(1,0)==2);
                REQUIRE(z.ReadMatrix(0)(0,1)==3);
                REQUIRE(z.ReadMatrix(0)(1,1)==4);
            }
        }
        WHEN("Tensor z is padding x with 0 at top, bottom, left and right by 1 respectively") {
            Tensor<float> z = x.Padding(std::vector<int>{1,1,1,1}, 0);

            THEN("z has been surrounded by 0") {
                REQUIRE(z.ReadShape() == std::vector{1,4,4});
                REQUIRE(z.ReadMatrix(0).block(1,1,2,2) == x.ReadMatrix(0));
                REQUIRE(z.ReadMatrix(0)(0,0)==0);
                REQUIRE(z.ReadMatrix(0)(0,3)==0);
                REQUIRE(z.ReadMatrix(0)(3,0)==0);
                REQUIRE(z.ReadMatrix(0)(3,3)==0);
            }
        }
    }
}

