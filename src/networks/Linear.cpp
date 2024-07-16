#include "networks/Linear.h"
#include "networks/LSTM.h"
#include "networks/Utils.h"
#include <iostream>
Linear::Linear(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim) {}

void Linear::load_weights(const std::string &prefix) {
    W = readCSV(prefix + "_weight.csv");
    b = readCSV(prefix + "_bias.csv");
}

Eigen::MatrixXd Linear::forward(const Eigen::MatrixXd &input) {
           
    return (W * input).colwise() + b;
}
