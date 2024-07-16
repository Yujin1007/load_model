#include "networks/LayerNorm.h"
#include "networks/LSTM.h"
#include "networks/Utils.h"
#include <iostream>

LayerNorm::LayerNorm(int dim, double eps) : dim(dim), eps(eps) {}

void LayerNorm::load_weights(const std::string &prefix)
{
    gamma = readCSV(prefix + "_weight.csv");
    beta = readCSV(prefix + "_bias.csv");
}

Eigen::MatrixXd LayerNorm::forward(const Eigen::MatrixXd &input)
{
    Eigen::VectorXd mean = input.colwise().mean();

    Eigen::MatrixXd centered = input.rowwise() - mean.transpose();

    // Calculate variance for each feature (column)
    Eigen::VectorXd variance = centered.array().square().colwise().mean();
    Eigen::VectorXd stddev = (variance.array() + eps).sqrt();
    // Eigen::MatrixXd normalized = centered.array().rowwise() / stddev.transpose().array();
    Eigen::VectorXd normalized = centered.array().rowwise() / stddev.transpose().array();
    // std::cout<<"normalized:"<<normalized.cols()<<"x"<<normalized.rows()<<" gamma::"<<gamma.cols()<<"x"<<gamma.rows()<<std::endl;

    normalized = normalized.array() * gamma.array();
    normalized = normalized.array() + beta.array();
    // std::cout << normalized.transpose() << std::endl;
    Eigen::MatrixXd normalized_matrix = normalized;
    return normalized_matrix;
}