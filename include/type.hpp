#ifndef SKY_INFER_DATA_TYPE
#define SKY_INFER_DATA_TYPE





namespace sky_infer {


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
        DeConv
    };



}





#endif