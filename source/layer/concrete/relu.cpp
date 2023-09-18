#include "layer/concrete/relu.hpp"

namespace sky_infer {

 //   ReLU::ReLU() : type_(LayerType::ReLU) {};


    void LayerReLU::Forward(Operator *opt) {
        CHECK(opt->inputs_.size()==1 && opt->outputs_.size()==1) << "failed to execute relu; relu allows only one input and one output";
        Operand* input = opt->inputs_[0];
        Operand* output = opt->outputs_[0];

        CHECK(input->data_.size()==output->data_.size() && input->data_[0].ReadShape()==output->data_[0].ReadShape()) << "failed to execute relu; input and output have different dimensions";

        for(int batch=0; batch < input->data_.size(); batch++) {
            for(int channel=0; channel <input->data_[0].ReadShape()[0]; channel++)
                for(int row=0; row<input->data_[0].ReadShape()[1]; row++)
                    for(int col=0; col<input->data_[0].ReadShape()[2]; col++) {
                        float value = input->data_[batch].ReadMatrix(channel)(row,col);
                        output->data_[batch].WriteMatrix(channel)(row,col) = value > 0 ? value:0;
                    }
                        // input->data_[batch].ReadMatrix(channel)(row,col)>0.f?
        }

    }
}