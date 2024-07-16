#ifndef ACTOR_H
#define ACTOR_H

#include "networks/LSTM.h"
#include "networks/NormalizedMLP.h"
#include "networks/MLP.h"
#include <eigen3/Eigen/Dense>
#include <string>

class ActorR {
public:
    ActorR(int state_dim, int lstm_hidden_dim, int mlp_hidden_dim, int output_dim);
    void load_weights(const std::string &directory);
    Eigen::VectorXd forward(const Eigen::MatrixXd &input);

private:
    LSTM *lstm_feature_extractor;
    NormalizedMLP *net;
};

class ActorF {
public:
    ActorF(int state_dim, int lstm_hidden_dim, int mlp_hidden_dim, int output_dim);
    void load_weights(const std::string &directory);
    Eigen::VectorXd forward(const Eigen::MatrixXd &input);

private:
    LSTM *lstm_feature_extractor;
    MLP *net;
};

#endif // ACTOR_H
