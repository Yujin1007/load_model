#ifndef NORMALIZEDMLP_H
#define NORMALIZEDMLP_H

#include "networks/Linear.h"
#include "networks/LayerNorm.h"
#include "networks/Utils.h"
#include <eigen3/Eigen/Dense>
#include <string>

class NormalizedMLP {
public:
    NormalizedMLP(int input_dim, std::array<int,2> hidden_dim, int output_dim);
    void load_weights(const std::string &directory);
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

private:
    Linear *layer1, *layer2, *layer3;
    LayerNorm *norm1, *norm2;
};

#endif // NORMALIZEDMLP_H
