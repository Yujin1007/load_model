#ifndef MLP_H
#define MLP_H

#include "networks/Linear.h"
#include "networks/Utils.h"
#include <eigen3/Eigen/Dense>
#include <string>

class MLP {
public:
    MLP(int input_dim, int hidden_dim, int output_dim);
    void load_weights(const std::string &directory);
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

private:
    Linear *layer1, *layer2, *layer3;
};

#endif 
