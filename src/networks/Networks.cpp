#include "networks/Networks.h"
#include "networks/LSTM.h"
#include "networks/NormalizedMLP.h"
#include "networks/MLP.h"
#include <iostream>
#include <chrono>


ActorR::ActorR(int state_dim, int lstm_hidden_dim, int mlp_hidden_dim, int output_dim) {
    lstm_feature_extractor = new LSTM(state_dim, lstm_hidden_dim);
    net = new NormalizedMLP(lstm_hidden_dim, mlp_hidden_dim, output_dim);
}


void ActorR::load_weights(const std::string &directory) {
    lstm_feature_extractor->load_weights(directory + "lstm_feature_extractor_");
    net->load_weights(directory);
}

Eigen::VectorXd ActorR::forward(const Eigen::MatrixXd &input) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    Eigen::MatrixXd lstm_output = lstm_feature_extractor->forward(input);

    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
    std::cout << "LSTM 걸리는 시간(초) : " << sec.count() *1000<<"ms"<< std::endl;

    start = std::chrono::system_clock::now();
    Eigen::MatrixXd last_output = lstm_output.col(lstm_output.cols() - 1); // Take the last output of the LSTM
    Eigen::MatrixXd log_output = net->forward(last_output);
    Eigen::VectorXd output = log_output.topRows(2).array().tanh();
    sec = std::chrono::system_clock::now() - start;
    std::cout << "MLP 걸리는 시간(초) : " << sec.count() *1000<<"ms"<< std::endl;

    return output;
}

ActorF::ActorF(int state_dim, int lstm_hidden_dim, int mlp_hidden_dim, int output_dim) {
    lstm_feature_extractor = new LSTM(state_dim, lstm_hidden_dim);
    net = new MLP(lstm_hidden_dim, mlp_hidden_dim, output_dim);
}


void ActorF::load_weights(const std::string &directory) {
    lstm_feature_extractor->load_weights(directory + "lstm_feature_extractor_");
    net->load_weights(directory);
}

Eigen::VectorXd ActorF::forward(const Eigen::MatrixXd &input) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    Eigen::MatrixXd lstm_output = lstm_feature_extractor->forward(input);

    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
    std::cout << "LSTM 걸리는 시간(초) : " << sec.count() *1000<<"ms"<< std::endl;


    start = std::chrono::system_clock::now();

    Eigen::MatrixXd last_output = lstm_output.col(lstm_output.cols() - 1); // Take the last output of the LSTM


    Eigen::MatrixXd log_output = net->forward(last_output);
    Eigen::VectorXd output = log_output.topRows(1).array().tanh();

    sec = std::chrono::system_clock::now() - start;
    std::cout << "MLP 걸리는 시간(초) : " << sec.count() *1000<<"ms"<< std::endl;

    return output;
}


Classifier::Classifier(int state_dim, int output_dim) {
    net = new MLPs(state_dim, output_dim);
}


void Classifier::load_weights(const std::string &directory) {
    net->load_weights(directory);
}

Eigen::VectorXd Classifier::forward(const Eigen::MatrixXd &input) {
    Eigen::MatrixXd output = net->forward(input);
    return output.array();
}


