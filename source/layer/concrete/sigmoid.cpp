#include "layer/concrete/sigmoid.hpp"


namespace sky_infer {
    LayerSigmoid::LayerSigmoid(std::string name, std::shared_ptr<Batchf> input, std::shared_ptr<Batchf> output) : type_(LayerType::Sigmoid), name_(std::move(name)), input_(std::move(input)), output_(std::move(output)) {};


    void LayerSigmoid::Forward() {

        check_(input_->size() ==output_->size()) << "failed to execute sigmoid; input and output have different batch sizes";

        for(int tensor=0; tensor < input_->size(); tensor++) {
            Tensor<float>& in = input_->at(tensor);
            Tensor<float>& out = output_->at(tensor);
            check_(in.Channels()==out.Channels() && in.Rows()==out.Rows() && in.Cols()==out.Cols()) << "failed to execute sigmoid; input tensor and output tensor have different shapes";
            for(int channel=0; channel <in.Channels(); channel++) {
                out.WriteMatrix(channel) = 1 / (1 + (-1.f * in.ReadMatrix(channel).array()).exp());
            }
        }
    }
}