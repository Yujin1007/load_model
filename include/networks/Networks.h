#ifndef NETWORKS_H
#define NETWORKS_H

#include "networks/LSTM.h"
#include "networks/NormalizedMLP.h"
#include "networks/MLP.h"
#include <eigen3/Eigen/Dense>
#include <string>

class ActorR {
public:
    ActorR(int state_dim, int lstm_hidden_dim, std::array<int, 2> mlp_hidden_dim, int output_dim);
    void load_weights(const std::string &directory);
    Eigen::VectorXd forward(const Eigen::MatrixXd &input);

private:
    LSTM *lstm_feature_extractor;
    NormalizedMLP *net;
};

class ActorF {
public:
    ActorF(int state_dim, int lstm_hidden_dim, std::array<int, 2> mlp_hidden_dim, int output_dim);
    void load_weights(const std::string &directory);
    Eigen::VectorXd forward(const Eigen::MatrixXd &input);

private:
    LSTM *lstm_feature_extractor;
    MLP *net;
};

class Actor {
public:
    Actor(int state_dim, std::array<int, 2> mlp_hidden_dim, int output_dim);
    void load_weights(const std::string &directory);
    Eigen::VectorXd forward(const Eigen::MatrixXd &input);

private:
    MLP *net;
};

class Classifier {
public:
    Classifier(int state_dim, int output_dim);
    void load_weights(const std::string &directory);
    Eigen::VectorXd forward(const Eigen::MatrixXd &input);

private:
    MLPs *net;
};


#endif // ACTOR_H
