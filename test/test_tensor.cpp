#include "tensor/tensor.hpp"
#include <catch2/catch_test_macros.hpp>





SCENARIO("tensor can be operated properly", "[tensor]") {

    using namespace nova_infer;

    GIVEN("two tensors x, y with some items") {
        set_multi_sink();
        std::vector<float> data1 = {1, 1, 1, 1};
        std::vector<float> data2 = {1, 2, 3, 4};

        Tensor<float> x(std::vector<int>{1, 2, 2}, data1);
        Tensor<float> y(std::vector<int>{1, 2, 2}, data2);

        REQUIRE(y.ReadMatrix(0)(1,0)==3);
        REQUIRE(y.ReadMatrix(0)(0,1)==2);
        REQUIRE(y.Max()==4);
        REQUIRE(y.Min()==1);

        WHEN("tensor z = x + y") {
            Tensor<float> z = x + y;

            THEN("z equals adding coefficients from x and y correspondingly") {
                REQUIRE(z.Channels() == x.Channels());
                REQUIRE(z.Rows() == x.Rows());
                REQUIRE(z.Cols() == x.Cols());
                REQUIRE(z.ReadMatrix(0)(0,0)==2);
                REQUIRE(z.ReadMatrix(0)(1,0)==4);
                REQUIRE(z.ReadMatrix(0)(0,1)==3);
                REQUIRE(z.ReadMatrix(0)(1,1)==5);
            }
        }
        WHEN("tensor z = x * y") {
            Tensor<float> z = x * y;

            THEN("z follows the rule of matrix dot product") {
                REQUIRE(z.Channels() == x.Channels());
                REQUIRE(z.Rows() == x.Rows());
                REQUIRE(z.Cols() == x.Cols());
                REQUIRE(z.ReadMatrix(0)(0,0)==4);
                REQUIRE(z.ReadMatrix(0)(1,0)==4);
                REQUIRE(z.ReadMatrix(0)(0,1)==6);
                REQUIRE(z.ReadMatrix(0)(1,1)==6);
            }
        }
        WHEN("tensor z = x % y") {
            Tensor<float> z = x % y;

            THEN("z equals multiplying coefficients from x and y correspondingly") {
                REQUIRE(z.Channels() == x.Channels());
                REQUIRE(z.Rows() == x.Rows());
                REQUIRE(z.Cols() == x.Cols());
                REQUIRE(z.ReadMatrix(0)(0,0)==1);
                REQUIRE(z.ReadMatrix(0)(1,0)==3);
                REQUIRE(z.ReadMatrix(0)(0,1)==2);
                REQUIRE(z.ReadMatrix(0)(1,1)==4);
            }
        }
        WHEN("x and y swap") {
            x.Swap(y);

            THEN("z equals multiplying coefficients from x and y correspondingly") {
                REQUIRE(x.ReadMatrix(0)(0,0)==1);
                REQUIRE(x.ReadMatrix(0)(1,0)==3);
                REQUIRE(x.ReadMatrix(0)(0,1)==2);
                REQUIRE(x.ReadMatrix(0)(1,1)==4);
                x.Swap(y);
            }
        }
        WHEN("Tensor z is padding x with 0 at top, bottom, left and right by 1 respectively") {
            Tensor<float> z = x.Padding(std::vector<int>{1,1,1,1}, 0);

            THEN("z has been surrounded by 0") {
                REQUIRE(z.Channels() == 1);
                REQUIRE(z.Rows() == 4);
                REQUIRE(z.Cols() == 4);
                REQUIRE(z.ReadMatrix(0).block(1,1,2,2) == x.ReadMatrix(0));
                REQUIRE(z.ReadMatrix(0)(0,0)==0);
                REQUIRE(z.ReadMatrix(0)(0,3)==0);
                REQUIRE(z.ReadMatrix(0)(3,0)==0);
                REQUIRE(z.ReadMatrix(0)(3,3)==0);
            }
        }
        WHEN("Tensor y has flattened each matrix") {
            Tensor<float> z = y.Reshape({4});

            THEN("z has one matrix with 1 row and 4 cols") {
                REQUIRE(z.Channels() == 1);
                REQUIRE(z.Rows() == 1);
                REQUIRE(z.Cols() == 4);
                REQUIRE(z.ReadMatrix(0)(0,0)==1);
                REQUIRE(z.ReadMatrix(0)(0,1)==2);
                REQUIRE(z.ReadMatrix(0)(0,2)==3);
                REQUIRE(z.ReadMatrix(0)(0,3)==4);
            }
        }
        WHEN("Tensor y has flattened each matrix inplace") {
            y.ReshapeInplace({4});

            THEN("y has one matrix with 1 row and 4 cols") {
                REQUIRE(y.Channels() == 1);
                REQUIRE(y.Rows() == 1);
                REQUIRE(y.Cols() == 4);
                REQUIRE(y.ReadMatrix(0)(0,0)==1);
                REQUIRE(y.ReadMatrix(0)(0,1)==2);
                REQUIRE(y.ReadMatrix(0)(0,2)==3);
                REQUIRE(y.ReadMatrix(0)(0,3)==4);
            }
        }




    }
}

