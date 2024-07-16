#ifndef ACTOR_H
#define ACTOR_H

#include "networks/LSTM.h"
#include "networks/NormalizedMLP.h"
#include <eigen3/Eigen/Dense>
#include <string>

class Actor {
public:
    Actor(int state_dim, int lstm_hidden_dim, int mlp_hidden_dim, int output_dim);
    void load_weights(const std::string &directory);
    Eigen::VectorXd forward(const Eigen::MatrixXd &input);

private:
    LSTM *lstm_feature_extractor;
    NormalizedMLP *net;
};

#endif // ACTOR_H
