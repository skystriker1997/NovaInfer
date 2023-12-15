#include "layer/concrete/softmax.hpp"


namespace nova_infer {

    LayerSoftmax::LayerSoftmax(std::string_view name,
                               std::vector<std::string> input_name,
                               std::vector<std::string> output_name,
                               int dim) {
        type_ = LayerType::SoftMax;
        name_ = name;
        input_names_ = std::move(input_name);
        output_names_ = std::move(output_name);
        dim_ = dim;
    }


    void LayerSoftmax::Forward() {
        int dim = dim_;
        omp_set_num_threads(omp_get_num_procs());

        for(int tensor=0; tensor < input_->size(); tensor++) {
            Tensor<float> &in = input_->at(tensor);
            Tensor<float> &out = output_->at(tensor);
            check_(in.Channels()==out.Channels() && in.Rows()==out.Rows() && in.Cols()==out.Cols())
            << "failed to execute softmax; input tensor and output tensor have different shapes";
            int raw_dim_size;

            if(in.Channels()==1 && in.Rows()==1) {
                raw_dim_size = 1;
            } else if(in.Channels()==1) {
                raw_dim_size = 2;
            } else {
                raw_dim_size = 3;
            }

            if(dim<0) {         // python list obj might use reversed index
                dim += raw_dim_size;   // convert to the c++'s convention if in this scenario
            }

            if(raw_dim_size==1) {
                dim += 2;
            } else if(raw_dim_size==2) {
                dim++;
            }

            check_(dim>=0 && dim<=2) << "failed to execute softmax; the target dimension is out of range";

            if(dim==0) {
                for(int row=0; row<in.Rows(); row++) {
                    for(int col=0; col<in.Cols(); col++) {
                        float max_val = 0.f;
                        for(int channel=0; channel<in.Channels(); channel++) {
                            int val = in.ReadMatrix(channel)(row, col);
                            max_val = std::max<float>(val, max_val);
                        }
                        float sum_val = 0.f;
                        for(int channel=0; channel<in.Channels(); channel++) {
                            int val = in.ReadMatrix(channel)(row, col);
                            float tmp_val = std::exp(val - max_val);
                            out.WriteMatrix(channel)(row, col) = tmp_val;
                            sum_val += tmp_val;
                        }
                        for(int channel=0; channel<in.Channels(); channel++) {
                            int val = out.ReadMatrix(channel)(row, col);
                            out.WriteMatrix(channel)(row, col) = val / sum_val;
                        }
                    }
                }
            } else if(dim==1) {
                for(int channel=0; channel<in.Channels(); channel++) {
                    for(int col=0; col<in.Cols(); col++) {
                        float max_val = 0.f;
                        for(int row=0; row<in.Rows(); row++) {
                            int val = in.ReadMatrix(channel)(row, col);
                            max_val = std::max<float>(val, max_val);
                        }
                        float sum_val = 0.f;
                        for(int row=0; row<in.Rows(); row++) {
                            int val = in.ReadMatrix(channel)(row, col);
                            float tmp_val = std::exp(val - max_val);
                            out.WriteMatrix(channel)(row, col) = tmp_val;
                            sum_val += tmp_val;
                        }
                        for(int row=0; row<in.Rows(); row++) {
                            int val = out.ReadMatrix(channel)(row, col);
                            out.WriteMatrix(channel)(row, col) = val / sum_val;
                        }
                    }
                }
            } else {
                for(int channel=0; channel<in.Channels(); channel++) {
                    for(int row=0; row<in.Rows(); row++) {
                        float max_val = 0.f;
                        for (int col=0; col<in.Rows(); col++) {
                            int val = in.ReadMatrix(channel)(row, col);
                            max_val = std::max<float>(val, max_val);
                        }
                        float sum_val = 0.f;
                        for(int col=0; col<in.Rows(); col++) {
                            int val = in.ReadMatrix(channel)(row, col);
                            float tmp_val = std::exp(val - max_val);
                            out.WriteMatrix(channel)(row, col) = tmp_val;
                            sum_val += tmp_val;
                        }
                        for(int col=0; col<in.Rows(); col++) {
                            int val = out.ReadMatrix(channel)(row, col);
                            out.WriteMatrix(channel)(row, col) = val / sum_val;
                        }
                    }
                }
            }
        }
    }


    std::shared_ptr<LayerSoftmax> MakeLayerSoftmax(pnnx::Operator *opt) {
        Check check;
        check(opt->inputs.size()==1) << "failed to create layer softmax; only accept one tensor as input";
        check(opt->outputs.size()==1) << "failed to create layer softmax; only produce one tensor as output";

        auto dim = opt->params.find("dim");
        check(dim != opt->params.end()) << "failed to create layer softmax; cannot find parameter dim";

        std::vector<std::string> input_name = {opt->inputs[0]->name};
        std::vector<std::string> output_name = {opt->outputs[0]->name};

        return std::make_shared<LayerSoftmax>(opt->name, std::move(input_name), std::move(output_name), dim->second.i);
    };


}