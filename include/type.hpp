#ifndef NOVA_INFER_DATA_TYPE
#define NOVA_INFER_DATA_TYPE





namespace nova_infer {


    enum LayerType {
        Dummy,
        ReLU,
        Expression,
        Linear,
        SoftMax,
        MaxPooling,
        Flatten,
        Sigmoid,
        Conv,
        DeConv,
        Cat
    };



}





#endif