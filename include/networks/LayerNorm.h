#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <eigen3/Eigen/Dense>
#include <string>

class LayerNorm {
public:
    LayerNorm(int dim);
    void load_weights(const std::string &prefix);
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

private:
    int dim;
    Eigen::VectorXd gamma;
    Eigen::VectorXd beta;
};

#endif // LAYERNORM_H
