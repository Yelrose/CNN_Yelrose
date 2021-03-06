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
    cout << shape.size() << endl;
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
    vector<Blob<double > * > label;

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

    Blob<double > * l = new Blob<double > (2,1,10);
    double * data = l -> mutable_data();
    for(int i = 0;i < 20;i ++) {
        data[i] = 0;
    }
    data[10 + 3] = 1;
    data[4] = 1;
    label.push_back(l);


    Layer<double> conv1 = Layer<double > ();
    Layer<double> pool1 = Layer<double > ();
    Layer<double> fully = Layer<double > ();
    Layer<double> sigmoid = Layer<double > ();
    Layer<double> relu = Layer<double > ();
    Layer<double > loss = Layer<double > ();


    conv1.set_convolution_mode(2,2,1,2);
    pool1.set_pooling_mode(2,2,2);
    fully.set_fully_mode(pool1_output_reshape,output_blobs);
    loss.set_softmax_with_cross_entropy_mode();

    cout << INFO << " Operation begin" << endl;
    cout <<"Convolution" << endl;
    conv1.convolution_forward(data_blobs,conv1_output);
    cout <<"Pooling " << endl;
    pool1.pooling_forward(conv1_output,pool1_output);

    cout <<"Fully" << endl;
    fully.fully_forward(pool1_output_reshape,output_blobs);

    cout <<"Sigmoid " << endl;
    sigmoid.sigmoid_forward(output_blobs,output_blobs_sigmoid);
    cout << "ReLU " <<endl;
    relu.ReLU_forward(output_blobs,output_blobs_relu);
    cout <<"entropy" << endl;
    loss.softmax_with_cross_entropy(output_blobs_sigmoid,label);
    cout <<"relu backward" << endl;
    relu.sigmoid_backward(output_blobs,output_blobs_sigmoid);

    cout <<"fully backward" << endl;
    fully.fully_backward(pool1_output_reshape,output_blobs,0.01);
    cout <<"pooling backward" << endl;
    pool1.pooling_backward(conv1_output,pool1_output);

    cout <<"conv backward" << endl;
    conv1.convolution_backward(data_blobs,conv1_output,0.01);






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
        cout << output_blobs_relu[i] -> shape().size() <<endl;
        print<double >(output_blobs_relu[i]);
    }



    //vector<Blob<double > * > data_blobs;
    //vector<Blob<double > * > conv1_output;
    //vector<Blob<double > * > pool1_output;
    //vector<Blob<double > * > pool1_output_reshape;
    //vector<Blob<double > * > output_blobs;
    //vector<Blob<double > * > output_blobs_sigmoid;
    //vector<Blob<double > * > output_blobs_relu;
    //vector<Blob<double > * > label;
    for(int i = 0;i < data_blobs.size() ;i ++) {
        delete data_blobs[i];
    }
    for(int i = 0 ;i < conv1_output.size();i ++) {
        delete conv1_output[i];
    }
    for(int i = 0;i < pool1_output.size();i ++) {
        delete pool1_output[i];
    }
    for(int i = 0;i < pool1_output_reshape.size();i ++) {
        delete pool1_output_reshape[i];
    }
    for(int i = 0;i < output_blobs.size();i ++) {
        delete output_blobs[i];
    }
    for(int i = 0;i < output_blobs_sigmoid.size();i ++) {
        delete output_blobs_sigmoid[i];
    }
    for(int i = 0;i < output_blobs_relu.size();i ++) {
        delete output_blobs_relu[i];
    }
    for(int i = 0;i < label.size();i ++) {
        delete label[i];
    }














    return 0;
}
