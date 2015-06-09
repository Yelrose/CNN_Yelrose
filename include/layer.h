#ifndef LAYER_H
#define LAYER_H
#include "debug.h"
#include "blob.h"
#include <iostream>
#include <vector>
using namespace std;

template <typename Dtype>
class Layer {
    public:
        Layer(){
        }
        void set_convolution_mode(int height,int width,int stride,int feature_map) {
            conv_height = height;
            conv_width = width;
            conv_stride = stride;
            conv_feature = feature_map;
            blobs_ = vector<Blob<Dtype> *> (conv_feature);
            for(int i = 0;i < blobs_.size();i ++) {
                blobs_[i] = new Blob<Dtype > (1,height,width);
                blobs_[i] -> random();

            }
        }
        void convolution_forward(const vector<Blob<Dtype> *>&bottom,const vector<Blob<Dtype>* > &top) {
            for(int i = 0;i < top.size();i ++){
                top[i] -> set_zero(); } for(int i = 0;i < blobs_.size();i ++) {
                const Dtype * weight_ =  blobs_[i] -> data();
                for(int j = 0;j < bottom.size();j ++) {
                    const Dtype * bottom_ = bottom[j] -> data();
                    Dtype * top_ = top[i] -> mutable_data();
                    convolution_forward_(weight_,bottom_,top_,bottom[j] -> shape(),top[i] -> shape());
                }
            }
        }
        void convolution_forward_(const Dtype * weight,const Dtype* bottom,Dtype * top,const vector<int > &bottom_shape,const vector<int > & top_shape) {
            int top_shift1 = top_shape[1] * top_shape[2];
            int top_shift2 = top_shape[2];
            int bottom_shift1 = bottom_shape[1] * bottom_shape[2];
            int bottom_shift2 = bottom_shape[2];
            for(int num = 0; num < top_shape[0];num ++) {
                for(int i = 0;i < bottom_shape[1] - conv_height + conv_stride;i += conv_stride) {
                    for(int j = 0;j < bottom_shape[2] - conv_width + conv_stride;j += conv_stride) {
                        for(int h = 0; h < conv_height;h ++) {
                            for(int w = 0;w < conv_width;w ++) {
                                int ii = i / conv_stride;
                                int jj = i / conv_stride;
                                top[num*top_shift1 + ii*top_shift2 +jj] +=
                                    bottom[num* bottom_shift1 + (i+h)* bottom_shift2 + j+w]
                                    * weight[h * conv_width + w];
                            }
                        }
                    }
                }
            }
        }
        void convolution_backward(const vector<Blob<Dtype> *>&bottom,const vector<Blob<Dtype>* > &top,Dtype learningRate) {
            int batch_size = bottom[0] -> shape()[0];
            for(int i = 0;i < bottom.size();i ++) {
                Dtype * bottom_diff_ = bottom[j] -> mutable_diff();
                memset(bottom_diff_,0,sizeof(bottom_diff_))
            }
            for(int i = 0;i < blobs_.size();i ++) {
                Dtype * blobs_diff_ = blobs[i].mutable_diff();
                memset(blobs_diff_,0,sizeof(blobs_diff_));
            }

            for(int i = 0;i < blobs_.size();i ++) {
                const Dtype* weight_ = blobs_[i] -> data();
                Dtype * weight_diff = blobs_[i] -> mutable_diff();
                for(int j = 0;j < bottom.size();j ++) {
                    const Dtype * bottom_ = bottom[j] -> data();
                    const Dtype* top_ = top[i] -> data();
                    Dtype * bottom_diff = bottom[j] -> mutable_diff();
                    Dtype * top_diff = top[i] -> diff();
                    // backpropagation
                    // bottom_diff = s
                    const vector<int > & top_shape = top[i] -> shape();
                    const vector<int > & bottom_shape = bottom[i] -> shape();
                    int top_shift1 = top_shape[1] * top_shape[2];
                    int top_shift2 = top_shape[2];
                    int bottom_shift1 = bottom_shape[1] * bottom_shape[2];
                    int bottom_shift2 = bottom_shape[2];
                    for(int num = 0; num < top_shape[0];num ++) {
                        for(int bh;bh < bottom_shape[1] - conv_height +conv_stride;bh += conv_stride) {
                            for(int bw;bw < bottom_shape[2] - conv_width + conv_stride;bw += conv_stride) {
                                for(int h = 0;h < conv_height;h ++) {
                                    for(int w = 0;w < conv_width;w ++) {
                                        th = bh / conv_height;
                                        tw = bw / conv_width;
                                        // bottom layer backpropagation
                                        bottom_diff[num*bottom_shift1 + bh*bottom_shift2 + bw] +=
                                            top_diff[num*top_shift1 + th * top_shift2 + tw]*
                                            weight_[h*conv_width + w];
                                        weight_diff[h*conv_width + w] +=
                                            bottom_[num*bottom_shift1 + bh*bottom_shift2 + bw] *
                                            top_diff[num*top_shift1 + th * top_shift2 + tw];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Update weight
            for(int i = 0;i < blobs_.size();i ++) {
                Dtype * weight_  = blobs_[i] -> mutable_data();
                const Dtype * weight_diff = blobs_[i] -> diff();
                for(int j = 0;j < conv_height * conv_width;j ++) {
                    weight_[j] = weight_[j] - learningRate/ batch_size* weight_diff[j];
                }
            }
        }



        void (int height,int width,int stride,int feature_map) {





    private:
        vector<Blob<Dtype >* > blobs_;
        int conv_height;
        int conv_width;
        int conv_stride;
        int conv_feature;
};


#endif
