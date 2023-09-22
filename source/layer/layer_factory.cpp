#include "layer/layer_factory.hpp"



namespace sky_infer {


    Layer* LayerFactory::GetLayer(sky_infer::LayerType type) {
        switch (type) {
            case LayerType::Input: {
                if(layers_.find(LayerType::Input) == layers_.end())
                    layers_.insert({type, std::make_unique<LayerInput>()});

                return layers_.at(type).get();
            }

            case LayerType::Output: {
                if(layers_.find(LayerType::Output) == layers_.end())
                    layers_.insert({type, std::make_unique<LayerOutput>()});

                return layers_.at(type).get();
            }

            case LayerType::ReLU: {
                if(layers_.find(LayerType::ReLU) == layers_.end())
                    layers_.insert({type, std::make_unique<LayerReLU>()});

                return layers_.at(type).get();
            }

            case LayerType::Expression: {
                if(layers_.find(LayerType::Expression) == layers_.end())
                    layers_.insert({type, std::make_unique<LayerExpression>()});

                return layers_.at(type).get();
            }

            case LayerType::MaxPooling: {
                if(layers_.find(LayerType::MaxPooling) == layers_.end())
                    layers_.insert({type, std::make_unique<LayerMaxpooling>()});

                return layers_.at(type).get();
            }

            case LayerType::Linear: {
                // TODO
                return nullptr;
            }
            case LayerType::SoftMax: {
                // TODO
                return nullptr;
            }
            default:
                LOG(ERROR) << "failed to retrieve layer; unsupported layer type: " << type;
        }
    }


}