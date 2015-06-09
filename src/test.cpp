#include "debug.h"
#include "layer.h"
#include "blob.h"
#include <iostream>
#include <vector>
using namespace std;


void print(const double * p, const vector<int > & shape) {
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
    vector<Blob<double > * > bottom;
    vector<Blob<double > * > top;
    vector<Blob<double > * > top2;
    Layer<double> a = Layer<double > ();
    Layer<double> b = Layer<double > ();
    a.set_convolution_mode(2,2,1,2);
    b.set_pooling_mode(2,2,2);
    Blob<double > * data [5];
    data[0] = new Blob<double >(2,5,5);
    double * p = data[0] -> mutable_data();
    for (int i = 0;i < data[0] -> size();i ++) {
        p[i] = i + 1;
    }
    bottom.push_back(data[0]);
    data[1] = new Blob<double >(2,4,4);
    top.push_back(data[1]);
    data[2] = new Blob<double >(2,4,4);
    top.push_back(data[2]);
    data[3] = new Blob<double >(2,2,2);
    top2.push_back(data[3]);
    data[4] = new Blob<double > (2,2,2);
    top2.push_back(data[4]);

    cerr << INFO << endl;
    a.convolution_forward(bottom,top);
    cerr << INFO << endl;
    b.pooling_forward(top,top2);
    cerr << INFO"Done" << endl;

    cout << "data layer" << endl;
    print(data[0] -> data(),data[0] -> shape());

    cout << "first feature map" << endl;
    const double * p1 = data[1] -> data();
    const vector<int >& p1_shape = data[1] -> shape();
    print(p1,p1_shape);

    cout << "second feature map" << endl;
    const double * p2 = data[2] -> data();
    const vector<int > & p2_shape = data[2] -> shape();
    print(p2,p2_shape);


    cout << "pooling layer" << endl;
    cout << "pooling 1" << endl;
    print(data[3]->data(),data[3] -> shape());
    cout << "pooling 2" << endl;
    print(data[4]->data(),data[4] -> shape());







    return 0;
}
