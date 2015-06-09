#ifndef BLOB_H
#define BLOB_H
#include <vector>
#include <cstring>
#include <iostream>
#include "debug.h"
using namespace std;

template <typename Dtype>
class Blob {
    public:
        Blob():data_(),diff_(),count_(0){

        }
        Blob(const int num,const int height,const int width) {
            shape_ = std::vector<int >(3);
            shape_[0] = num;
            shape_[1] = height;
            shape_[2] = width;
            count_ = 1;
            for(int i = 0;i < shape_.size();i ++) {
                count_ *= shape_[i];
            }
            if(count_ > 0) {
                data_ = new Dtype [count_];
                diff_ = new Dtype [count_];
            }
            else {
                cerr << INFO << " count_ == 0" << endl;
            }
        }
        ~Blob() {
            if(count_ > 0) {
                delete [] data_;
                delete [] diff_; }
        }
        void random() {
            for(int i = 0;i < count_;i ++) {
                data_[i] = 1;
            }
        }
        void Reshape(const int num,const int height,const int width) {
            int count = 1;
            vector<int > shape = std::vector<int >(3);
            shape[0] = num;
            shape[1] = height;
            shape[2] = width;
            for(int i = 0;i < shape.size();i ++) {
                count *= shape[i];
            }
            if(count != count_) {
                cerr << INFO << "count "<<count << " != count_ "<<count_<<endl;
                shape_ = shape;
            }
        }
        const std::vector<int > & shape() const {
            return shape_;
        }
        Dtype* mutable_data() {
            return data_;
        }
        Dtype* mutable_diff() {
            return diff_;
        }
        const Dtype * data() const {
            return data_;
        }
        const Dtype * diff() const  {
            return diff_;
        }
        void set_zero() {
            memset(data_,0,sizeof(data_));
            memset(diff_,0,sizeof(diff_));
        }
        int size() {
            return count_;
        }
    private:
        Dtype * data_;
        Dtype * diff_;
        std::vector<int > shape_;
        int count_;
};

#endif
