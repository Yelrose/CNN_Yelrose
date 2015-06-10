#Convolution Neural Network

This project implement a convolution neural network (CNN ) for study. Motivated by the beauty of [Caffe](https://github.com/BVLC/caffe), I decided to build my own architecture of neural network and help myself to have a fully understand of how neural network works.

There are mainly two modules in this architecture.

* Blobs
* Layer

##Blobs
Blobs stored data of layer's weights or features. To protect the data from changes, we can call ***data()*** or ***diff()*** for const array. If we want to update the data, we can call ***mutable_data()*** or ***mutabel_diff()*** for returning changeable array.

##Layer
I have implemented the followed layers.

* Convolution layer
* Max-pooling layer
* Fully-connected layer
* ReLU layer
* Sigmoid layer

Each layer has ***forward*** and ***backward*** operation.



