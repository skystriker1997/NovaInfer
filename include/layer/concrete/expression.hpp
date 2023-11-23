#ifndef NOVA_INFER_LAYER_EXPRESSION
#define NOVA_INFER_LAYER_EXPRESSION

#include "layer/layer.hpp"


namespace nova_infer {


    class LayerExpression: public Layer {
    private:
        std::string name_;
        LayerType type_;
        Check check_;
        std::vector<std::string> input_names_;
        std::vector<std::string> output_name_;

        std::vector<std::shared_ptr<Batchf>> inputs_;
        std::shared_ptr<Batchf> output_;

        std::string expression_;
        std::vector<std::string> token_vector_;

        void Parse();


    public:
        LayerExpression(std::string name,
                        std::vector<std::string> input_names, std::vector<std::string> output_name,
                        std::string expression);

        void AssignInputs(std::vector<std::shared_ptr<Batchf>> inputs) {inputs_ = inputs;};

        void AssignOutput(std::shared_ptr<Batchf> output) override {output_ = output;};

        void Forward() override;

        ~LayerExpression() override = default;
    };


    std::shared_ptr<LayerExpression> MakeLayerExpression(pnnx::Operator *opt);

}




#endif