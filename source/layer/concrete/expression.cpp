#include "layer/concrete/expression.hpp"



namespace sky_infer {

    LayerExpression::LayerExpression(std::string name, std::vector<std::shared_ptr<Batchf>> inputs,  std::shared_ptr<Batchf> output, std::string expression): type_(LayerType::Expression), name_(std::move(name)), inputs_(std::move(inputs)), output_(std::move(output)), expression_(std::move(expression)) {
        Parse();
    }

    void LayerExpression::Forward() {

      //  check_(outputs_.size()==1) << "failed to execute expression; only support one output";

        int n = token_vector_.size();

        std::vector<Batchf> tmp_batches(n);
        std::vector<std::shared_ptr<Batchf>> batches(n);


        for (int i = n - 1; i > 0; i--) {
            if (token_vector_[i] != "add" && token_vector_[i] != "mul") {
                int input_index;
                try {
                    input_index = std::stoi(token_vector_[i]);  // Convert string to int
                    //    std::cout << "Converted value: " << intValue << std::endl;
                } catch (const std::invalid_argument &e) {
                    check_(false) << "failed to execute expression; invalid input index: " + std::string(e.what());
                } catch (const std::out_of_range &e) {
                    check_(false) << "failed to execute expression; input index out of int range: " + std::string(e.what());
                }
                check_(input_index >= 0 && input_index < inputs_.size()) << "failed to execute expression; input index out of vector range: " + std::to_string(input_index);
                batches[i] = inputs_[input_index];
            }
        }

        batches[0] = output_;

        for(int i=n-1; i>=0; i--) {

            if(batches[i])
                continue;

            int input1_index = i+1;
            while(!batches[input1_index] && tmp_batches[input1_index].empty())
                input1_index++;
            int input2_index = input1_index+1;
            while(!batches[input2_index] && tmp_batches[input2_index].empty())
                input2_index++;

            bool input1_tmp = !batches[input1_index];
            bool input2_tmp = !batches[input2_index];

            Batchf & input1 = input1_tmp ? tmp_batches[input1_index] : *batches[input1_index];
            Batchf & input2 = input2_tmp ? tmp_batches[input2_index] : *batches[input2_index];


            check_(input1.size() == input2.size()) << "failed to execute expression; illegal add or mul: mismatching batch sizes";

            if(i) {
                tmp_batches[i].resize(input1.size());
                if(token_vector_[i] == "add")
                    for(int j=0; j<input1.size(); j++) {
                        check_(input1[j].Channels()==input2[j].Channels() && input1[j].Rows()==input2[j].Rows() && input1[j].Cols()==input2[j].Cols()) << "failed to execute expression; tensors should have same shape for addition";
                        tmp_batches[i][j] = input1[j] + input2[j];
                    }

                else
                    for(int j=0; j<input1.size(); j++) {
                        check_(input1[j].Channels()==input2[j].Channels() && input1[j].Rows()==input2[j].Rows() && input1[j].Cols()==input2[j].Cols()) << "failed to execute expression; tensors should have same shape for coefficient-wise multiplication";
                        tmp_batches[i][j] = input1[j] % input2[j];
                    }
            } else {
                if(token_vector_[i] == "add")
                    for(int j=0; j<input1.size(); j++) {
                        check_(input1[j].Channels()==input2[j].Channels() && input1[j].Rows()==input2[j].Rows() && input1[j].Cols()==input2[j].Cols()) << "failed to execute expression; tensors should have same shape for addition";
                        check_(input1[j].Channels()==batches[0]->at(j).Channels() && input1[j].Rows()==batches[0]->at(j).Rows() && input1[j].Cols()==batches[0]->at(j).Cols()) << "failed to execute expression; tensors should have same shape for addition";
                        batches[0]->at(j) = input1[j] + input2[j];
                    }

                else
                    for(int j=0; j<input1.size(); j++) {
                        check_(input1[j].Channels()==input2[j].Channels() && input1[j].Rows()==input2[j].Rows() && input1[j].Cols()==input2[j].Cols()) << "failed to execute expression; tensors should have same shape for coefficient-wise multiplication";
                        check_(input1[j].Channels()==batches[0]->at(j).Channels() && input1[j].Rows()==batches[0]->at(j).Rows() && input1[j].Cols()==batches[0]->at(j).Cols()) << "failed to execute expression; tensors should have same shape for coefficient-wise multiplication";
                        batches[0]->at(j) = input1[j] % input2[j];
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





}