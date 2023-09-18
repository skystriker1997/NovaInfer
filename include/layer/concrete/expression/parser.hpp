#ifndef SKY_INFER_PARSER
#define SKY_INFER_PARSER

#include "operator.hpp"
#include "operand.hpp"
#include "tensor/tensor.hpp"
#include <vector>



namespace sky_infer {


    class ExpressionParser {

    public:
        std::vector<std::string> token_vector_;


        explicit ExpressionParser(const std::string& expression);

        ~ExpressionParser() = default;

    };



}




#endif