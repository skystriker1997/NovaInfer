#ifndef SKY_INFER_LAYER_FACTORY
#define SKY_INFER_LAYER_FACTORY



#include "layer/layer.hpp"

#include "layer/concrete/relu.hpp"
#include "layer/concrete/expression/expression.hpp"
#include "layer/concrete/input.hpp"
#include "layer/concrete/output.hpp"
#include "layer/concrete/maxpooling.hpp"


namespace sky_infer {

    class LayerFactory {

    private:
        std::map<LayerType, std::unique_ptr<Layer>> layers_;

    public:

        LayerFactory() = default;

        Layer* GetLayer(LayerType type);

        ~LayerFactory() = default;

    };


}


#endif