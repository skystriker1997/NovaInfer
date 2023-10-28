#include "layer/concrete/relu.hpp"

namespace sky_infer {

    LayerReLU::LayerReLU(std::string name, std::vector<std::string> input_name, std::vector<std::string> output_name)
    {
        type_ = LayerType::ReLU;
        name_ = std::move(name);
        input_name_ = std::move(input_name);
        output_name_ = std::move(output_name);
    };


    void LayerReLU::Forward() {

        for(int tensor=0; tensor < input_->size(); tensor++) {
            Tensor<float> &in = input_->at(tensor);
            Tensor<float> &out = output_->at(tensor);
            check_(in.Channels()==out.Channels() && in.Rows()==out.Rows() && in.Cols()==out.Cols()) <<
            "failed to execute relu; input tensor and output tensor have different shapes";

            for(int channel=0; channel < in.Channels(); channel++)
                out.WriteMatrix(channel) = in.ReadMatrix(channel).unaryExpr([](float i){return i > 0.f ? i:0.f;});
        }
    }

    std::shared_ptr<LayerReLU> MakeLayerReLU(pnnx::Operator *opt) {
        Check check;

        check(opt->inputnames.size()==1) << "failed to create layer relu; only accept one tensor as input";
        check(opt->outputs.size()==1) << "failed to create layer relu; only produce one tensor as output";

        std::vector<std::string> output_name = {opt->outputs[0]->name};

        return std::make_shared<LayerReLU>(std::move(opt->name), std::move(opt->inputnames), std::move(output_name));

    };

}