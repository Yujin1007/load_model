#include "networks/LayerNorm.h"
#include "networks/LSTM.h"
#include "networks/Utils.h"
#include <iostream>

LayerNorm::LayerNorm(int dim) : dim(dim) {}

void LayerNorm::load_weights(const std::string &prefix)
{
    gamma = readCSV(prefix + "_weight.csv");
    beta = readCSV(prefix + "_bias.csv");
}

Eigen::MatrixXd LayerNorm::forward(const Eigen::MatrixXd &input) {
    Eigen::VectorXd mean = input.colwise().mean();
    Eigen::MatrixXd centered = input.rowwise() - mean.transpose();
    Eigen::VectorXd variance = (centered.array().square().colwise().sum() / (input.rows() - 1)).transpose();

    Eigen::VectorXd stddev = (variance.array() + 1e-5).sqrt();

    Eigen::MatrixXd normalized = centered.array().rowwise() / stddev.transpose().array();

    if (gamma.size() == input.cols() && beta.size() == input.cols()) {
        normalized = normalized.array().rowwise() * gamma.transpose().array();
        normalized = normalized.array().rowwise() + beta.transpose().array();
    }

    return normalized;
}
// Eigen::MatrixXd LayerNorm::forward(const Eigen::MatrixXd &input) {
//     Eigen::VectorXd mean = input.colwise().mean();
//     std::cout << "Shape of m: (" << mean.size() << ", 1)\n";
//     Eigen::VectorXd variance = ((input.rowwise() - mean.transpose()).array().square().colwise().mean());
//     std::cout << "Shape of v: (" << variance.size() << ", 1)\n";

//     // Eigen::VectorXd normalized = (input.rowwise() - mean.transpose()).array().rowwise() / variance.sqrt().transpose().array();
//     Eigen::VectorXd normalized = (input.rowwise() - mean.transpose()).array().rowwise() / variance.array().sqrt().transpose().array();
//     std::cout << "Shape of n: (" << normalized.size() << ", 1)\n";
//     std::cout << "Shape of gamma: (" << gamma.size() << ", 1)\n";
//     std::cout << "Shape of beta: (" << beta.size() << ", 1)\n";

//     return (normalized.array().rowwise() * gamma.transpose().array()).matrix() + beta.transpose();
// }
