#ifndef SKY_INFER_DATA_TYPE
#define SKY_INFER_DATA_TYPE





namespace sky_infer {


//    enum OpdDataType {
//        OpdUnknown,
//        OpdFloat,
//    };
//
//    enum AttrDataType {
//        AttrUnknown,
//        AttrFloat,
//    };


//    enum ParamDataType {
//        ParamDummy,
//
//        ParamBool,
//
//        ParamInt,
//        ParamFloat,
//        ParamString,
//
//        ParamIntArray,
//        ParamFloatArray,
//        ParamStringArray
//    };


    enum LayerType {
        Dummy,
        Input,
        Output,
        ReLU,
        Expression,
        Linear,
        SoftMax,
        MaxPooling,
        Flatten,
        Sigmoid
    };


//    enum Tensor {
//        Dummy,
//        Mul
//    };


}





#endif