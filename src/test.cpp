#include "debug.h"
#include "layer.h"
#include "blob.h"
#include <iostream>
#include <vector>
using namespace std;


int main() {
    vector<Blob<double > * > bottom;
    vector<Blob<double > * > top;
    Layer<double> a = Layer<double > ();
    a.set_convolution_mode(2,2,1,2);
    Blob<double > * data [3];
    data[0] = new Blob<double >(2,2,2);
    double * p = data[0] -> mutable_data();
    for (int i = 0;i < 8;i ++) {
        p[i] = i;
    }
    bottom.push_back(data[0]);
    data[1] = new Blob<double >(2,1,1);
    top.push_back(data[1]);
    data[2] = new Blob<double >(2,1,1);
    top.push_back(data[2]);

    cerr << INFO << endl;
    a.convolution_forward(bottom,top);
    cerr << INFO"Done" << endl;
    for(int i = 0;i < 8;i ++) {
        cout << p[i] << " ";
    }
    cout << endl;
    const double * p1 = data[1] -> data();
    for(int i = 0;i < 2;i ++) {
        cout << p1[i] << " ";
    }
    cout << endl;
    const double * p2 = data[2] -> data();
    for(int i = 0;i < 2;i ++) {
        cout << p2[i] << " ";
    }
    cout << endl;





    return 0;
}
