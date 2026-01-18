# 0 Dependency Rust Neural Network on MNIST Dataset

## Overview
This project is a 0 dependency Neural Network written from scratch in Rust. It is currently setup to train and be tested on the MNIST dataset but the underlying network could be used in any other environment.

The current implementation has managed to achieve a 97.54% accuracy on the MNIST test data set.

## Currently Implemented

1. Matrix Module, with basic helper methods
2. Training Data module, used for converting datasets into something that the network can read
3. Forward Propagation
4. MSE Loss
5. Backpropagation to implement gradient descent and reduce network loss

## Running Training and Testing
1. go to https://github.com/phoebetronic/mnist and download mnist_test.csv and mnist_test.csv files (not folders), and place them inside /data folder in repo main directory
2. Run:
```
cargo run --release
```

## Recommended Settings
The settings used to achieve 97.54% were as follows:
```
Dimensions: 784x256x128x10
Alpha: 0.0005
Generations: 20
```