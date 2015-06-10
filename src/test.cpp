#include "debug.h"
#include "layer.h"
#include "blob.h"
#include <iostream>
#include <vector>
using namespace std;

template <typename Dtype>
void print(const Blob<Dtype > *tmp) {//const double * p, const vector<int > & shape) {
    const Dtype * p = tmp -> data();
    const vector<int > & shape = tmp -> shape();
    int shift1 = shape[1] * shape[2];
    int shift2 = shape[2];
    for(int num = 0;num < shape[0];num ++) {
        cout << "#"<<num <<" matrix" << endl;
        for(int h = 0;h < shape[1];h ++) {
            for(int w = 0;w < shape[2];w ++) {
                cout << p[num * shift1 + h * shift2 + w] << " ";
            }
            cout << endl;
        }

    }
}

int main() {
    vector<Blob<double > * > data_blobs;
    vector<Blob<double > * > conv1_output;
    vector<Blob<double > * > pool1_output;
    vector<Blob<double > * > pool1_output_reshape;
    vector<Blob<double > * > output_blobs;
    vector<Blob<double > * > output_blobs_sigmoid;
    vector<Blob<double > * > output_blobs_relu;

    cerr << INFO << " data_blobs init" << endl;

    for(int i = 0;i < 2;i ++) {
        Blob<double > * p = new Blob<double > (2,5,5);
        double * data = p -> mutable_data();
        for(int i = 0;i < p -> size();i ++) {
            data[i] =1.0 * (i + 1) / p->size();

        }
        data_blobs.push_back(p);
    }



    cerr << INFO << " conv1_ouput init" <<  endl;
    for(int i = 0;i < 2;i ++) {
        Blob<double > * p = new Blob<double > (2,4,4);
        double * data = p -> mutable_data();
        conv1_output.push_back(p);
    }

    cerr << INFO << " pool1_output & pool1_output_reshape init" <<  endl;
    for(int i = 0;i < 2;i ++) {
        Blob<double > * p = new Blob<double > (2,2,2);
        double * data = p -> mutable_data();
        Blob<double > * p2 = p ->Share_Reshape_Blob(2,1,4);
        pool1_output.push_back(p);
        pool1_output_reshape.push_back(p2);
    }

    cerr << INFO << " output_blobs init" << endl;
    Blob<double > * p = new Blob<double > (2,1,10);
    output_blobs.push_back(p);



    cerr << INFO << " output_blobs_sigmoid init" << endl;
    Blob<double > * p1 = new Blob<double > (2,1,10);
    output_blobs_sigmoid.push_back(p1);


    cerr << INFO << " output_blobs_relu init" << endl;
    Blob<double > * p2 = new Blob<double > (2,1,10);
    output_blobs_relu.push_back(p2);



    Layer<double> conv1 = Layer<double > ();
    Layer<double> pool1 = Layer<double > ();
    Layer<double> fully = Layer<double > ();
    Layer<double> sigmoid = Layer<double > ();
    Layer<double> relu = Layer<double > ();


    conv1.set_convolution_mode(2,2,1,2);
    pool1.set_pooling_mode(2,2,2);
    fully.set_fully_mode(pool1_output_reshape,output_blobs);


    conv1.convolution_forward(data_blobs,conv1_output);
    pool1.pooling_forward(conv1_output,pool1_output);
    fully.fully_forward(pool1_output_reshape,output_blobs);
    sigmoid.sigmoid_forward(output_blobs,output_blobs_sigmoid);
    relu.ReLU_forward(output_blobs,output_blobs_relu);


    cerr <<INFO << " data_blobs" << endl;
    for(int i = 0;i < data_blobs.size();i ++) {
        print<double >(data_blobs[i]);
    }

    cerr <<INFO << " conv1_output" << endl;
    for(int i = 0;i < conv1_output.size();i ++) {
        print<double >(conv1_output[i]);
    }


    cerr <<INFO << " pool1_output" << endl;
    for(int i = 0;i < pool1_output.size();i ++) {
        print<double >(pool1_output[i]);
    }


    cerr <<INFO << " pool1_output_reshape" << endl;
    for(int i = 0;i < pool1_output_reshape.size();i ++) {
        print<double >(pool1_output_reshape[i]);
    }

    cerr <<INFO << " output_blobs" << endl;
    for(int i = 0;i < output_blobs.size();i ++) {
        print<double >(output_blobs[i]);
    }


    cerr <<INFO << " output_blobs_sigmoid" << endl;
    for(int i = 0;i < output_blobs_sigmoid.size();i ++) {
        print<double >(output_blobs_sigmoid[i]);
    }


    cerr <<INFO << " output_blobs_relu" << endl;
    for(int i = 0;i < output_blobs_relu.size();i ++) {
        print<double >(output_blobs_relu[i]);
    }
















    return 0;
}
