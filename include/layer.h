#ifndef LAYER_H
#define LAYER_H
#include "debug.h"

enum layer_type ={
    CONVOLUTION,
    POOLING,
    DATA,
    LOSS
};
template <typename Dtype>
class Layer {
    public:
        void forward(const vector<Blob<Dtype> *>&bottom,const vector<Blob<Dtype>* > &top) {
        }
        void backward(const vector<Blob<Dtype> *>&bottom,const vector<Blob<Dtype>* > &top) {
        }
};


#endif
