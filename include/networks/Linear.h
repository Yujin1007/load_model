#ifndef LINEAR_H
#define LINEAR_H

#include <eigen3/Eigen/Dense>
#include <string>

class Linear {
public:
    Linear(int input_dim, int output_dim);
    void load_weights(const std::string &prefix);
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

private:
    int input_dim;
    int output_dim;
    Eigen::MatrixXd W;
    Eigen::VectorXd b;
};

#endif // LINEAR_H
