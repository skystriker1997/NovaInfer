#include "layer/concrete/expression/parser.hpp"

namespace sky_infer {
    ExpressionParser::ExpressionParser(const std::string &expression) {
        auto lptr = expression.begin();
        auto rptr = expression.begin();
        while (lptr != expression.end()) {
            char c = *lptr;
            if (c == '@') {
                do
                    rptr++;
                while (std::isdigit(*rptr));
                token_vector_.emplace_back(lptr + 1, rptr);
                lptr = rptr;
            } else if (c == 'a') {
                //  rptr++;
                CHECK(*(rptr++) == 'd' && *(rptr++) == 'd')
                                << "failed to parse expression; unidentified operator starting with: "
                                << std::string{lptr, rptr++};
                token_vector_.emplace_back("add");
                rptr++;
                lptr = rptr;
            } else if (c == 'm') {
                CHECK(*(rptr++) == 'u' && *(rptr++) == 'l')
                                << "failed to parse expression; unidentified operator starting with: "
                                << std::string{lptr, rptr++};
                token_vector_.emplace_back("mul");
                rptr++;
                lptr = rptr;
            } else if (c == ' ' || c == '(' || c == ')' || c == ',') {
                rptr++;
                lptr = rptr;
            } else {
                LOG(FATAL) << "failed to parse expression; unidentified item starting with: "
                           << std::string{lptr, lptr + 1};
            }
        }

//        operands_.resize(token_vector_.size());
//        for (int i = token_vector_.size() - 1; i > 0; i--) {
//            if (token_vector_[i] != "add" && token_vector_[i] != "mul") {
//                int opd_index;
//                try {
//                    opd_index = std::stoi(token_vector_[i]);  // Convert string to int
//                    //    std::cout << "Converted value: " << intValue << std::endl;
//                } catch (const std::invalid_argument &e) {
//                    LOG(FATAL) << "invalid opd_index: " << e.what() << std::endl;
//                } catch (const std::out_of_range &e) {
//                    LOG(FATAL) << "opd_index out of int range: " << e.what() << std::endl;
//                }
//                CHECK(opd_index >= 0 && opd_index < inputs.size()) << "opd_index out of vector range: " << opd_index;
//                operands_[i] = inputs[opd_index]->data_;
//            }
//        }
//        operands_[0] = outputs[0]->data_;
    }


//
//    ExpressionParser::~ExpressionParser() {
//        for (int i = 0; i < token_vector_.size(); i++) {
//            if (i && (token_vector_[i] == "add" || token_vector_[i] == "mul"))
//                delete operands_[i];
//        }
//    }
}