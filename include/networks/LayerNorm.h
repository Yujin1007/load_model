#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <eigen3/Eigen/Dense>
#include <string>

class LayerNorm {
public:
    LayerNorm(int dim, double eps=1e-05);
    void load_weights(const std::string &prefix);
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

private:
    int dim;
    double eps;
    Eigen::VectorXd gamma;
    Eigen::VectorXd beta;

};

#endif // LAYERNORM_H
