#include "layer/concrete/relu.hpp"

namespace sky_infer {

    LayerReLU::LayerReLU(std::shared_ptr<Batch<float>> input, std::shared_ptr<Batch<float>> output) : type_(LayerType::ReLU), input_(input), output_(output) {};


    void LayerReLU::Forward() {

        check_(input_->shape_==output_->shape_) << "failed to execute relu; input and output have different dimensions";

        for(int batch=0; batch < input_->shape_[0]; batch++) {
            for(int channel=0; channel <input_->shape_[1]; channel++)
                for(int row=0; row<input_->shape_[2]; row++)
                    for(int col=0; col<input_->shape_[3]; col++) {
                        float value = input_->data_[batch].ReadMatrix(channel)(row,col);
                        output_->data_[batch].WriteMatrix(channel)(row,col) = value > 0 ? value:0;
                    }
        }
    }
}