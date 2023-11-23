#include "layer/concrete/flatten.hpp"

namespace nova_infer {
    LayerFlatten::LayerFlatten(std::string name,
                               std::vector<std::string> input_name, std::vector<std::string> output_name,
                               int start_dim, int end_dim)
    {
        type_ = LayerType::Flatten;
        name_ = std::move(name);
        input_name_ = std::move(input_name);
        output_name_ = std::move(output_name);
        start_dim_ = start_dim;
        end_dim_ = end_dim;
    };


    void LayerFlatten::Forward() {

        int start = start_dim_ < 0 ? 4 + start_dim_ : start_dim_;
        int end = end_dim_ < 0 ? 4 + end_dim_ : end_dim_;

        check_(start < end) << "failed to execute flatten layer; the start dimension must be less than end dimension";
        check_(start >= 1 && end <= 3)
                << "failed to execute flatten layer; the start dimension cannot be less than 1 and the end dimension cannot be "
                   "greater than 3";

        if (start == 1 && end == 3) {
            omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
            for (int i = 0; i < input_->size(); i++) {
                Tensor<float> &in = input_->at(i);
                Tensor<float> &out = output_->at(i);
                int new_channel = 1;
                int new_row = 1;
                int new_col = in.Channels() * in.Rows() * in.Cols();
                check_(out.Channels()==new_channel && out.Rows()==new_row && out.Cols()==new_col)
                        << "failed to execute flatten layer; the shape of output tensor is not as expected";
                out = in.Reshape({new_channel, new_row, new_col});
            }
        } else if (start == 2 && end == 3) {
            omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
            for (int i = 0; i < input_->size(); i++) {
                Tensor<float> &in = input_->at(i);
                Tensor<float> &out = output_->at(i);
                int new_channel = 1;
                int new_row = in.Channels();
                int new_col = in.Rows() * in.Cols();
                check_(out.Channels()==new_channel && out.Rows()==new_row && out.Cols()==new_col)
                        << "failed to execute flatten layer; the shape of output tensor is not as expected";
                out = in.Reshape({new_channel, new_row, new_col});
            }
        } else {
            omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
            for (int i = 0; i < input_->size(); i++) {
                Tensor<float> &in = input_->at(i);
                Tensor<float> &out = output_->at(i);
                int new_channel = 1;
                int new_row = in.Channels() * in.Rows();
                int new_col = in.Cols();
                check_(out.Channels()==new_channel && out.Rows()==new_row && out.Cols()==new_col)
                        << "failed to execute flatten layer; the shape of output tensor is not as expected";
                out = in.Reshape({new_channel, new_row, new_col});
            }
        }
    }


    std::shared_ptr<LayerFlatten> MakeLayerFlatten(pnnx::Operator *opt) {
        Check check;
        check(opt->inputnames.size()==1) << "failed to create layer flatten; only accept one tensor as input";
        check(opt->outputs.size()==1) << "failed to create layer flatten; only produce one tensor as output";

        auto start_dim = opt->params.find("start_dim");
        check(start_dim != opt->params.end()) << "failed to create flatten layer; cannot find parameter start_dim";

        auto end_dim = opt->params.find("end_dim");
        check(end_dim != opt->params.end()) << "failed to create flatten layer; cannot find parameter end_dim";

        std::vector<std::string> output_name = {opt->outputs[0]->name};

        return std::make_shared<LayerFlatten>(std::move(opt->name),
                                              std::move(opt->inputnames), std::move(output_name),
                                              start_dim->second.i, end_dim->second.i);
    };

}
