#include "layer/concrete/relu.hpp"

namespace sky_infer {

    LayerReLU::LayerReLU(std::string name, std::shared_ptr<Batchf> input, std::shared_ptr<Batchf> output) : type_(LayerType::ReLU), name_(std::move(name)), input_(std::move(input)), output_(std::move(output)) {};


    void LayerReLU::Forward() {

        check_(input_->size()==output_->size()) << "failed to execute relu; input batch and output batch have different number of tensors";

        for(int tensor=0; tensor < input_->size(); tensor++) {
            Tensor<float>& in = input_->at(tensor);
            Tensor<float>& out = output_->at(tensor);
            check_(in.Channels()==out.Channels() && in.Rows()==out.Rows() && in.Cols()==out.Cols()) << "failed to execute relu; input tensor and output tensor have different shapes";
            for(int channel=0; channel < in.Channels(); channel++)
                out.WriteMatrix(channel) = in.ReadMatrix(channel).unaryExpr([](float i){return i > 0.f ? i:0.f;});
        }
    }
}