#include "layer/concrete/expression.hpp"



namespace nova_infer {

    LayerExpression::LayerExpression(std::string name,
                                     std::vector<std::string> input_names, std::vector<std::string> output_name,
                                     std::string expression):
                                     type_(LayerType::Expression),
                                     name_(std::move(name)),
                                     input_names_(std::move(input_names)), output_name_(std::move(output_name)),
                                     expression_(std::move(expression))
    {
        Parse();
    }



    void LayerExpression::Forward() {

        int n = token_vector_.size();

        std::vector<Batchf> tmp_batches(n);
        std::vector<std::shared_ptr<Batchf>> batches(n);

        for (int op = n - 1; op > 0; op--) {
            if (token_vector_[op] != "add" && token_vector_[op] != "mul") {
                int input_index;
                try {
                    input_index = std::stoi(token_vector_[op]);  // Convert string to int
                } catch (const std::invalid_argument &e) {
                    check_(false) << "failed to execute expression; invalid input index: " + std::string(e.what());
                } catch (const std::out_of_range &e) {
                    check_(false) << "failed to execute expression; input index out of int range: " + std::string(e.what());
                }
                check_(input_index >= 0 && input_index < inputs_.size()) << "failed to execute expression; input index out of vector range: " + std::to_string(input_index);
                batches[op] = inputs_[input_index];
            }
        }

        batches[0] = output_;

        for (int op=n-1; op>=0; op--) {

            if(batches[op])
                continue;

            int input1_index = op+1;
            while(!batches[input1_index] && tmp_batches[input1_index].empty())
                input1_index++;
            int input2_index = input1_index+1;
            while(!batches[input2_index] && tmp_batches[input2_index].empty())
                input2_index++;

            bool input1_tmp = !batches[input1_index];
            bool input2_tmp = !batches[input2_index];

            Batchf &input1 = input1_tmp ? tmp_batches[input1_index] : *batches[input1_index];
            Batchf &input2 = input2_tmp ? tmp_batches[input2_index] : *batches[input2_index];

            check_(input1.size() == input2.size()) << "failed to execute expression; illegal add or mul: mismatching batch sizes";

            if(op) {
                tmp_batches[op].resize(input1.size());
                if(token_vector_[op] == "add")
                    for(int t=0; t<input1.size(); t++) {
                        check_(input1[t].Channels()==input2[t].Channels() && input1[t].Rows()==input2[t].Rows() && input1[t].Cols()==input2[t].Cols()) << "failed to execute expression; tensors should have same shape for addition";
                        tmp_batches[op][t] = input1[t] + input2[t];
                    }

                else
                    for(int t=0; t<input1.size(); t++) {
                        check_(input1[t].Channels()==input2[t].Channels() && input1[t].Rows()==input2[t].Rows() && input1[t].Cols()==input2[t].Cols()) << "failed to execute expression; tensors should have same shape for coefficient-wise multiplication";
                        tmp_batches[op][t] = input1[t] % input2[t];
                    }
            } else {
                if(token_vector_[op] == "add")
                    for(int t=0; t<input1.size(); t++) {
                        check_(input1[t].Channels()==input2[t].Channels() && input1[t].Rows()==input2[t].Rows() && input1[t].Cols()==input2[t].Cols()) << "failed to execute expression; tensors should have same shape for addition";
                        check_(input1[t].Channels()==batches[0]->at(t).Channels() && input1[t].Rows()==batches[0]->at(t).Rows() && input1[t].Cols()==batches[0]->at(t).Cols()) << "failed to execute expression; tensors should have same shape for addition";
                        batches[0]->at(t) = input1[t] + input2[t];
                    }

                else
                    for(int t=0; t<input1.size(); t++) {
                        check_(input1[t].Channels()==input2[t].Channels() && input1[t].Rows()==input2[t].Rows() && input1[t].Cols()==input2[t].Cols()) << "failed to execute expression; tensors should have same shape for coefficient-wise multiplication";
                        check_(input1[t].Channels()==batches[0]->at(t).Channels() && input1[t].Rows()==batches[0]->at(t).Rows() && input1[t].Cols()==batches[0]->at(t).Cols()) << "failed to execute expression; tensors should have same shape for coefficient-wise multiplication";
                        batches[0]->at(t) = input1[t] % input2[t];
                    }
            }

            if(input1_tmp)
                tmp_batches[input1_index] = Batchf();
            else
                batches[input1_index].reset();

            if(input2_tmp)
                tmp_batches[input2_index] = Batchf();
            else
                batches[input2_index].reset();

        }
    }



    void LayerExpression::Parse() {
        auto lptr = expression_.begin();
        auto rptr = expression_.begin();
        while (lptr != expression_.end()) {
            char c = *lptr;
            if (c == '@') {
                do
                    rptr++;
                while (std::isdigit(*rptr));
                token_vector_.emplace_back(lptr + 1, rptr);
                lptr = rptr;
            } else if (c == 'a') {
                //  rptr++;
                check_(*(rptr++) == 'd' && *(rptr++) == 'd')
                        << "failed to parse expression; unidentified operator starting with: "
                        + std::string{lptr, rptr++};
                token_vector_.emplace_back("add");
                rptr++;
                lptr = rptr;
            } else if (c == 'm') {
                check_(*(rptr++) == 'u' && *(rptr++) == 'l')
                        << "failed to parse expression; unidentified operator starting with: "
                        + std::string{lptr, rptr++};
                token_vector_.emplace_back("mul");
                rptr++;
                lptr = rptr;
            } else if (c == ' ' || c == '(' || c == ')' || c == ',') {
                rptr++;
                lptr = rptr;
            } else {
                check_(false) << "failed to parse expression; unidentified item starting with: "
                           + std::string{lptr, lptr + 1};
            }
        }
    }

    std::shared_ptr<LayerExpression> MakeLayerExpression(pnnx::Operator *opt) {
        Check check;
        check(!opt->inputnames.empty()) << "failed to create layer expression; input should contain at least one tensor";
        check(opt->outputs.size()==1) << "failed to create layer expression; only produce one tensor as output";

        auto expr = opt->params.find("expr");
        check(expr != opt->params.end()) << "failed to create layer expression; cannot find parameter expression";

        std::vector<std::string> output_name = {opt->outputs[0]->name};

        return std::make_shared<LayerExpression>(std::move(opt->name), std::move(opt->inputnames), std::move(output_name), std::move(expr->second.s));
    };

}