#include "layer/concrete/expression/expression.hpp"


namespace sky_infer {
    void LayerExpression::Forward(sky_infer::Operator *opt) {
      //  const std::string& expression = opt->params_.at("expr"))->
        ExpressionParser parser(dynamic_cast<ParameterString*>(opt->params_.at("expr").get())->value_);

        int n = parser.token_vector_.size();


        std::vector<std::vector<Tensor<float>>> hypercubes(n);
        std::vector<Operand*> opds(n);


             //   operands_.resize(token_vector_.size());
        for (int i = n - 1; i > 0; i--) {
            if (parser.token_vector_[i] != "add" && parser.token_vector_[i] != "mul") {
                int opd_index;
                try {
                    opd_index = std::stoi(parser.token_vector_[i]);  // Convert string to int
                    //    std::cout << "Converted value: " << intValue << std::endl;
                } catch (const std::invalid_argument &e) {
                    LOG(FATAL) << "invalid opd_index: " << e.what() << std::endl;
                } catch (const std::out_of_range &e) {
                    LOG(FATAL) << "opd_index out of int range: " << e.what() << std::endl;
                }
                CHECK(opd_index >= 0 && opd_index < opt->inputs_.size()) << "opd_index out of vector range: " << opd_index;
                opds[i] = opt->inputs_[opd_index];
            }
        }

        opds[0] = opt->outputs_[0];

        for(int i=n-1; i>=0; i--) {

            if(opds[i])
                continue;

            int input1_index = i+1;
            while(!opds[input1_index] && hypercubes[input1_index].empty())
                input1_index++;
            int input2_index = input1_index+1;
            while(!opds[input2_index] && hypercubes[input2_index].empty())
                input2_index++;

            bool input1_tmp = !opds[input1_index];
            bool input2_tmp = !opds[input2_index];

            std::vector<Tensor<float>>& input1 = input1_tmp?hypercubes[input1_index]:opds[input1_index]->data_;
            std::vector<Tensor<float>>& input2 = input2_tmp?hypercubes[input2_index]:opds[input2_index]->data_;


            CHECK(input1.size() == input2.size()) << "failed to execute add or mul of operands; mismatching batch sizes";

//            int max_channel = input1[0].GetShape()[0]>input2[0].GetShape()[0]?input1[0].GetShape()[0]:input2[0].GetShape()[0];
//            int max_row = input1[0].GetShape()[1]>input2[0].GetShape()[1]?input1[0].GetShape()[1]:input2[0].GetShape()[1];
//            int max_col = input1[0].GetShape()[2]>input2[0].GetShape()[2]?input1[0].GetShape()[2]:input2[0].GetShape()[2];

            if(i) {
//                auto t = std::vector<Tensor<float>>(input1.size(), Tensor<float>(std::vector<int>{max_channel, max_row, max_col}));
                hypercubes[i].resize(input1.size());
                if(parser.token_vector_[i] == "add")
                    for(int j=0; j<input1.size(); j++)
                        hypercubes[i][j] = input1[j] + input2[j];
                else
                    for(int j=0; j<input1.size(); j++)
                        hypercubes[i][j] = input1[j] % input2[j];
            } else {
                if(parser.token_vector_[i] == "add")
                    for(int j=0; j<input1.size(); j++)
                        opds[0]->data_[j] = input1[j] + input2[j];
                else
                    for(int j=0; j<input1.size(); j++)
                        opds[0]->data_[j] = input1[j] % input2[j];
            }

            if(input1_tmp)
                hypercubes[input1_index].clear();
            else
                opds[input1_index] = nullptr;

            if(input2_tmp)
                hypercubes[input2_index].clear();
            else
                opds[input2_index] = nullptr;

        }

    }
}