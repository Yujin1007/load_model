#include "networks/NormalizedMLP.h"
#include "networks/Linear.h"
#include "networks/LayerNorm.h"
#include "networks/LSTM.h"
#include <iostream>

NormalizedMLP::NormalizedMLP(int input_dim, std::array<int,2> hidden_dim, int output_dim) {
    layer1 = new Linear(input_dim, hidden_dim[0]);
    norm1 = new LayerNorm(hidden_dim[0]);
    layer2 = new Linear(hidden_dim[0], hidden_dim[1]);
    norm2 = new LayerNorm(hidden_dim[1]);
    layer3 = new Linear(hidden_dim[1], output_dim);
}

void NormalizedMLP::load_weights(const std::string &directory) {
    layer1->load_weights(directory + "net_model_0");
    norm1->load_weights(directory + "net_model_1");
    layer2->load_weights(directory + "net_model_3");
    norm2->load_weights(directory + "net_model_4");
    layer3->load_weights(directory + "net_model_6");
}

Eigen::MatrixXd NormalizedMLP::forward(const Eigen::MatrixXd &input) {
    Eigen::MatrixXd x = relu(norm1->forward(layer1->forward(input)));
    x = relu(norm2->forward(layer2->forward(x)));
    return layer3->forward(x);
}
