#include "networks/MLP.h"
#include "networks/Linear.h"
#include "networks/LSTM.h"
#include <iostream>

MLP::MLP(int input_dim, int hidden_dim, int output_dim) {
    layer1 = new Linear(input_dim, hidden_dim);
    layer2 = new Linear(hidden_dim, hidden_dim);
    layer3 = new Linear(hidden_dim, output_dim);
}

void MLP::load_weights(const std::string &directory) {
    layer1->load_weights(directory + "net_model_0");
    layer2->load_weights(directory + "net_model_2");
    layer3->load_weights(directory + "net_model_4");
}

Eigen::MatrixXd MLP::forward(const Eigen::MatrixXd &input) {
    Eigen::MatrixXd x = relu(layer1->forward(input));
    x = relu(layer2->forward(x));
    return layer3->forward(x);
}

MLPs::MLPs(int input_dim, int output_dim) {
    layer1 = new Linear(input_dim, 256);
    layer2 = new Linear(256, 1024);
    layer3 = new Linear(1024, 1024);
    layer4 = new Linear(1024, 256);
    layer5 = new Linear(256, output_dim);
}

void MLPs::load_weights(const std::string &directory) {
    layer1->load_weights(directory + "fc1");
    layer2->load_weights(directory + "fc2");
    layer3->load_weights(directory + "fc3");
    layer4->load_weights(directory + "fc_");
    layer5->load_weights(directory + "fc4");
}

Eigen::MatrixXd MLPs::forward(const Eigen::MatrixXd &input) {
    Eigen::MatrixXd x = relu(layer1->forward(input));
    x = relu(layer2->forward(x));
    x = relu(layer3->forward(x));
    x = relu(layer4->forward(x));
    
    return layer5->forward(x);
}