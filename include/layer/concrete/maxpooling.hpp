#ifndef NOVA_INFER_LAYER_MAXPOOLING
#define NOVA_INFER_LAYER_MAXPOOLING


#include "layer/layer.hpp"

namespace nova_infer {

    class LayerMaxpooling: public Layer {

    private:
        std::string name_;
        LayerType type_;
        Check check_;
        std::vector<std::string> input_name_;
        std::vector<std::string> output_name_;

        std::shared_ptr<Batchf> input_;
        std::shared_ptr<Batchf> output_;

        int stride_h_;
        int stride_w_;
        int padding_h_;
        int padding_w_;
        int kernel_h_;
        int kernel_w_;


    public:
        LayerMaxpooling(std::string name,
                        std::vector<std::string> input_name, std::vector<std::string> output_name,
                        int stride_h, int stride_w,
                        int padding_h, int padding_w,
                        int kernel_h, int kernel_w);

        void AssignInput(std::shared_ptr<Batchf> input) override {input_ = input;};

        void AssignOutput(std::shared_ptr<Batchf> output) override {output_ = output;};


        void Forward() override;


        ~LayerMaxpooling() override = default;

    };


    std::shared_ptr<LayerMaxpooling> MakeLayerMaxpooling(pnnx::Operator *opt);
}





#endif