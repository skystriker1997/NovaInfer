#include "layer/concrete/sigmoid.hpp"


namespace nova_infer {
    LayerSigmoid::LayerSigmoid(std::string_view name, std::vector<std::string> input_name, std::vector<std::string> output_name)
    {
        type_ = LayerType::Sigmoid;
        name_ = name;
        input_names_ = std::move(input_name);
        output_names_ = std::move(output_name);
    };


    void LayerSigmoid::Forward() {

        omp_set_num_threads(omp_get_num_procs());

        for(int tensor=0; tensor < input_->size(); tensor++) {
            Tensor<float> &in = input_->at(tensor);
            Tensor<float> &out = output_->at(tensor);
            check_(in.Channels()==out.Channels() && in.Rows()==out.Rows() && in.Cols()==out.Cols())
            << "failed to execute sigmoid; input tensor and output tensor have different shapes";
            for(int channel=0; channel <in.Channels(); channel++) {
                out.WriteMatrix(channel) = 1 / (1 + (-1.f * in.ReadMatrix(channel).array()).exp());
            }
        }
    }

    std::shared_ptr<LayerSigmoid> MakeLayerSigmoid(pnnx::Operator *opt) {
        Check check;
        check(opt->inputs.size()==1) << "failed to create layer sigmoid; only accept one tensor as input";
        check(opt->outputs.size()==1) << "failed to create layer sigmoid; only produce one tensor as output";

        std::vector<std::string> input_name = {opt->inputs[0]->name};
        std::vector<std::string> output_name = {opt->outputs[0]->name};

        return std::make_shared<LayerSigmoid>(opt->name, std::move(input_name), std::move(output_name));
    };

}