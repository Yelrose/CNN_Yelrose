#ifndef LAYER_H
#define LAYER_H
#include "debug.h"
#include "blob.h"
#include <iostream>
using namespace std;

template <typename Dtype>
class Layer {
    public:
        Layer(){
        }
        void set_convolution_mode(int height,int weight,int stride,int feature_map) {
            conv_height = height;
            conv_weight = weight;
            conv_stride = stride;
            conv_feature = feature_map;
            blobs_ = vector<Blob<Dtype> *> (conv_feature);
            for(int i = 0;i < blobs_.size();i ++) {
                blobs_[i] = new Blob(1,1,height,weight)
            }
        }
        void forward(const vector<Blob<Dtype> *>&bottom,const vector<Blob<Dtype>* > &top) {


        }
        void backward(const vector<Blob<Dtype> *>&bottom,const vector<Blob<Dtype>* > &top) {

        }
    private:
        vector<Blob<Dtype >* > blobs_;
        int conv_height;
        int conv_weight;
        int conv_stride;
        int conv_feature;
};


#endif
