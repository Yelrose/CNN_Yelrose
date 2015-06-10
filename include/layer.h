#ifndef LAYER_H
#define LAYER_H
#include "debug.h"
#include "blob.h"
#include <iostream>
#include <vector>
#include <cmath>
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
                top[i] -> set_zero();
            }
            for(int i = 0;i < blobs_.size();i ++) {
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
                        int ii = i / conv_stride;
                        int jj = j / conv_stride;
                        for(int h = 0; h < conv_height;h ++) {
                            for(int w = 0;w < conv_width;w ++) {

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
                Dtype * bottom_diff_ = bottom[i] -> mutable_diff();
                memset(bottom_diff_,0,sizeof(bottom_diff_));
            }
            for(int i = 0;i < blobs_.size();i ++) {
                Dtype * blobs_diff_ = blobs_[i].mutable_diff();
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
                                int th = bh / conv_height;
                                int tw = bw / conv_width;
                                for(int h = 0;h < conv_height;h ++) {
                                    for(int w = 0;w < conv_width;w ++) {
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


        /*
         * TODO poolng layer implement
         */
        void set_pooling_mode(int height,int width,int stride) {
            pool_height = height;
            pool_width = width;
            pool_stride = stride;
        }

        void pooling_forward(const vector<Blob<Dtype> *>&bottom,const vector<Blob<Dtype>* > &top) {
            if(bottom.size() != top.size()) cerr << INFO <<" bottom size " << bottom.size() << " != top size " <<  top.size() << endl;
            for(int i = 0;i < bottom.size();i ++) {
                const Dtype * bottom_ =  bottom[i] -> data();
                Dtype * top_ =  top[i] -> mutable_data();
                const vector<int > & bottom_shape =  bottom[i]->shape();
                const vector<int > & top_shape = top[i] -> shape();
                int top_shift1 = top_shape[1] * top_shape[2];
                int top_shift2 = top_shape[2];
                int bottom_shift1 = bottom_shape[1] * bottom_shape[2];
                int bottom_shift2 = bottom_shape[2];
                for(int num = 0;num < top_shape[0];num ++) {
                    for(int bh = 0;bh < bottom_shape[1] - pool_height+pool_stride;bh += pool_stride) {
                        for(int bw = 0;bw < bottom_shape[2] - pool_width+pool_stride;bw += pool_stride) {
                            bool first = false;
                            int th =  bh / pool_stride;
                            int tw = bw / pool_stride;
                            for(int h = 0;h < pool_height;h ++) {
                                for(int w = 0;w < pool_width;w ++) {
                                    if(first ){
                                        top_[num * top_shift1 + th * top_shift2 + tw] = bottom_[num* bottom_shift1 + (bh+h) * bottom_shift2 + (bw+w)];
                                        first = false;
                                    }
                                    top_[num * top_shift1 + th * top_shift2 + tw] = max(
                                            top_[num * top_shift1 + th * top_shift2 + tw] ,
                                            bottom_[num* bottom_shift1 +  (bh +h) * bottom_shift2 + (bw+w)]
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        void pooling_backward(const vector<Blob<Dtype> *>&bottom,const vector<Blob<Dtype>* > &top) {
            if(bottom.size() != top.size()) cerr << INFO <<" bottom size " << bottom.size() << " != top size " <<  top.size() << endl;
            for(int i = 0;i < bottom.size();i ++) {
                const Dtype * bottom_ =  bottom[i] -> data();
                const Dtype * top_ =  top[i] -> data();
                const Dtype * top_diff = top[i] -> diff();
                Dtype * bottom_diff = bottom[i] -> mutable_diff();
                const vector<int > & bottom_shape =  bottom[i]->shape();
                const vector<int > & top_shape = top[i] -> shape();
                int top_shift1 = top_shape[1] * top_shape[2];
                int top_shift2 = top_shape[2];
                int bottom_shift1 = bottom_shape[1] * bottom_shape[2];
                int bottom_shift2 = bottom_shape[2];
                for(int num = 0;num < top_shape[0];num ++) {
                    for(int bh = 0;bh < bottom_shape[1] - pool_height+ pool_stride;bh += pool_stride) {
                        for(int bw = 0;bw < bottom_shape[2] - pool_width+pool_stride;bw += pool_stride) {
                            bool first = false;
                            int th =  bh / pool_stride;
                            int tw = bw / pool_stride;
                            for(int h = 0;h < pool_height;h ++) {
                                for(int w = 0;w < pool_width;w ++) {
                                    if(top_[num * top_shift1 + th * top_shift2 + tw] ==bottom_[num* bottom_shift1  + bh * bottom_shift2 + bw]) {
                                        bottom_diff[num*bottom_shift1 + bh * bottom_shift2 + bw] = top_diff[num*top_shift1 + th* top_shift2 + tw];
                                    }
                                    else {
                                        bottom_diff[num*bottom_shift1 + bh * bottom_shift2 + bw]  = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }



        void set_fully_mode(const vector<Blob<Dtype>* > bottom,const vector<Blob<Dtype>*> top) {
            int height = 0;
            for (int i = 0;i < bottom.size();i ++) {
                height += bottom[i] -> shape()[2];
            }
            int width =  top[0] -> shape()[2];
            Blob<Dtype> * p  = new Blob<Dtype>(1,height,width);
            p -> random();
            blobs_.clear();
            blobs_.push_back(p);
        }

        void fully_forward(const vector<Blob<Dtype>* > bottom,const vector<Blob<Dtype>*> top) {
            top[0] -> set_zero();
            int tot_num =  top[0]-> shape()[0];
            int tot_width = top[0] -> shape()[2];
            Dtype * top_ = top[0] -> mutable_data();
            const Dtype * weight_ =  blobs_[0] -> data();
            int tot_height = 0;
            for(int i = 0;i < bottom.size();i ++) {
                int height = bottom[i] -> shape()[2];
                tot_height += height;
                const Dtype * bottom_ =  bottom[i] -> data();
                for (int num = 0;num < tot_num;num ++) {
                    for (int h = 0;h < height;h ++) {
                        for (int w = 0;w < tot_width;w ++) {
                            top_[num * tot_width + w] += bottom_[num * height + h] * weight_[(tot_height + h) * tot_width + w];
                        }
                    }
                }
            }
        }

        void fully_backward(const vector<Blob<Dtype>* > bottom,const vector<Blob<Dtype>*> top,Dtype learningRate) {
            for(int i = 0;i < bottom.size();i ++) {
                Dtype * p = bottom[i] -> mutable_data();
                memset(p,0,sizeof(p));
            }
            for(int i = 0;i < blobs_.size();i ++) {
                Dtype * p = blobs_[i] -> mutable_data();
                memset(p,0,sizeof(p));
            }

            // back propagation
            int tot_num =  top[0]-> shape()[0];
            int tot_width = top[0] -> shape()[2];
            const Dtype * top_diff = top[0] -> diff();
            const Dtype * weight_ =  blobs_[0] -> data();
            Dtype * weight_diff =blobs_[0] -> mutable_diff();
            int tot_height = 0;
            for(int i = 0;i < bottom.size();i ++) {
                int height = bottom[i] -> shape()[2];
                tot_height += height;
                const Dtype * bottom_ =  bottom[i] -> data();
                Dtype * bottom_diff = bottom[i] -> mutable_diff();
                for (int num = 0;num < tot_num;num ++) {
                    for (int h = 0;h < height;h ++) {
                        for (int w = 0;w < tot_width;w ++) {
                            //top_[num * tot_width + w] += bottom_[num * height + h] * weight_[(tot_height + h) * tot_width + w];
                            bottom_diff[num*height+h] += top_diff[num*tot_width+w] *weight_[(tot_height + h) * tot_width+w];
                            weight_diff[(tot_height+h) * tot_width+1]  += top_diff[num* tot_width + w] * bottom_[num *height +h];
                        }
                    }
                }
            }

            Dtype * new_weight = blobs_[0] -> mutable_data();
            // Update weight
            for(int i = 0;i < tot_width * tot_height;i ++) {
                new_weight[i] = new_weight[i] - learningRate / tot_num * weight_diff[i];
            }

        }


        void set_softmax_mode() {

            return 0;
        }
        Dtype sigmoid(Dtype x) {
            return 1. / (1 + exp(-x));
        }

        void sigmoid_forward(const vector<Blob<Dtype>* > bottom,const vector<Blob<Dtype>*> top) {
            if(bottom.size() != top.size()) {
                cerr << INFO << " sigmoid layer bottom size "<<bottom.size() << " != top size "  << top.size() << endl;
            }
            for(int i = 0;i < bottom.size();i ++) {
                if(bottom[i] -> shape() != top[i] -> shape()) {
                    cerr<<INFO << " bottom shape " << bottom[i]->shape()[0] << "x" << bottom[i] -> shape()[1] << "x" << bottom[i] -> shape()[2]  <<
                        " not equal " << "top shape " << top[i] -> shape()[0] << "x" << top[i] -> shape()[1] << "x" << top[i] -> shape()[2] << endl;
                }
                const Dtype * bottom_ = bottom[i] -> data();
                Dtype * top_ = top[i] -> mutable_data();
                for(int j = 0;j < bottom[i] -> size();j ++) {
                    top_[j] =  sigmoid(bottom_[j]);
                }
            }
        }


        void sigmoid_backward(const vector<Blob<Dtype>* > bottom,const vector<Blob<Dtype>*>top) {
            for(int i = 0;i < bottom.size();i ++) {
                const Dtype * bottom_ = bottom[i] -> data();
                const Dtype * top_diff = top[i] -> diff();
                const Dtype * top_ = top[i] -> data();
                Dtype * bottom_diff = bottom -> mutable_diff();
                for(int j = 0;j < top[i] -> size();j ++) {
                    const Dtype sigmoid_x = top_[j];
                    bottom_diff[j] = top_diff[j] * top_[j] * (1 - top_[j]);
                }
            }
        }



        void set_ReLU_mode() {

        }


        void ReLU_forward(const vector<Blob<Dtype>* > bottom,const vector<Blob<Dtype>*> top) {
            if(bottom.size() != top.size()) {
                cerr << INFO << " ReLU layer bottom size "<<bottom.size() << " != top size "  << top.size() << endl;
            }
            for(int i = 0;i < bottom.size();i ++) {
                if(bottom[i] -> shape() != top[i] -> shape()) {
                    cerr<<INFO << " bottom shape " << bottom[i]->shape()[0] << "x" << bottom[i] -> shape()[1] << "x" << bottom[i] -> shape()[2]  <<
                        " not equal " << "top shape " << top[i] -> shape()[0] << "x" << top[i] -> shape()[1] << "x" << top[i] -> shape()[2] << endl;
                }
                const Dtype * bottom_ = bottom[i] -> data();
                Dtype * top_ = top[i] -> mutable_data();
                for(int j = 0;j < bottom[i] -> size();j ++) {
                    top_[j] =  max(bottom_[j],(Dtype)0.0);
                }
            }
        }


        void ReLU_backward(const vector<Blob<Dtype>* > bottom,const vector<Blob<Dtype>*>top) {
            for(int i = 0;i < bottom.size();i ++) {
                const Dtype * bottom_ = bottom[i] -> data();
                const Dtype * top_diff = top[i] -> diff();
                const Dtype * top_ = top[i] -> data();
                Dtype * bottom_diff = bottom -> mutable_diff();
                for(int j = 0;j < top[i] -> size();j ++) {
                    bottom_diff[j] = top_diff[j] *( (bottom_[j] > 0)?1.0:0.0);
                }
            }
        }





    private:
        vector<Blob<Dtype >* > blobs_;
        int conv_height;
        int conv_width;
        int conv_stride;
        int conv_feature;
        int pool_height;
        int pool_width;
        int pool_stride;
};


#endif
