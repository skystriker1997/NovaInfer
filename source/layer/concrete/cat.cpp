#include "layer/concrete/cat.hpp"


namespace nova_infer {
    LayerCat::LayerCat(std::string name, std::vector<std::string> input_names, std::vector<std::string> output_name,
                       int dim) {
        type_ = LayerType::Cat;
        name_ = std::move(name);
        input_names_ = std::move(input_names);
        output_name_ = std::move(output_name);
        dim_ = dim;
    }


    void LayerCat::Forward() {
        int batch_size = inputs_[0]->size();


        for(auto &batch: inputs_) {
            check_(batch->size() == batch_size) << "failed to execute layer cat; all batches should have same batch size";
        }
        check_(batch_size == output_->size()) << "failed to execute layer cat; the output should have same batch size with the input";

        check_(dim_==1 || dim_==-3) << "failed to execute layer cat; only support the concatenation along channel";

#pragma omp parallel for
        for(int t=0; t<output_->size(); t++) {
            Tensor<float> &out = output_->at(t);
            int channels = out.Channels();
            int sum = 0;
            for(auto &batch: inputs_) {
                Tensor<float> &in = batch->at(t);
                sum += in.Channels();
                check_(in.Rows()==out.Rows() && in.Cols()==out.Cols()) << "failed to execute layer cat; tensors going to be concatenated should have same rows and cols";
            }
            check_(sum == channels) << "failed to execute layer cat; the output has mismatching number of channels with aggregated inputs";
            int start = 0;
            for(auto &batch: inputs_) {
                Tensor<float> &in = batch->at(t);
                for(int c=0; c<in.Channels(); c++) {
                    out.WriteMatrix(start+c) = in.ReadMatrix(c);
                }
                start += in.Channels();
            }
        }
    }



    std::shared_ptr<LayerCat> MakeLayerCat(pnnx::Operator *opt) {
        Check check;

        check(opt->outputs.size() == 1) << "failed to create layer cat; only produce one tensor as output";

        std::vector<std::string> output_name = {opt->outputs[0]->name};

        auto dim = opt->params.find("dim");
        check(dim != opt->params.end()) << "failed to create layer cat; cannot find parameter dim";

        return std::make_shared<LayerCat>(std::move(opt->name), std::move(opt->inputnames), std::move(output_name), dim->second.i);
    };

}