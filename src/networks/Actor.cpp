#include "networks/Actor.h"
#include "networks/LSTM.h"
#include "networks/NormalizedMLP.h"
#include <iostream>

Actor::Actor(int state_dim, int lstm_hidden_dim, int mlp_hidden_dim, int output_dim) {
    lstm_feature_extractor = new LSTM(state_dim, lstm_hidden_dim);
    net = new NormalizedMLP(lstm_hidden_dim, mlp_hidden_dim, output_dim);
}

void Actor::load_weights(const std::string &directory) {
    lstm_feature_extractor->load_weights(directory + "lstm_feature_extractor_");
    net->load_weights(directory);
}

Eigen::VectorXd Actor::forward(const Eigen::MatrixXd &input) {
    Eigen::MatrixXd lstm_output = lstm_feature_extractor->forward(input);
    Eigen::MatrixXd last_output = lstm_output.col(lstm_output.cols() - 1); // Take the last output of the LSTM
    std::cout<<"forward lstm"<<std::endl;
    // std::cout<<last_output<<std::endl;
    Eigen::MatrixXd log_output = net->forward(last_output);
    Eigen::VectorXd output = log_output.topRows(2).array().tanh();
    // Eigen::VectorXd output = log_output.array().tanh();
    std::cout<<output.transpose()<<std::endl;
    return output;
}
