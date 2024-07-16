#ifndef UTILS_H
#define UTILS_H

#include <eigen3/Eigen/Dense>
#include <string>

Eigen::MatrixXd readCSV(const std::string &path);
Eigen::MatrixXd relu(const Eigen::MatrixXd &x);
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &x);
Eigen::MatrixXd tanh(const Eigen::MatrixXd &x);

#endif // UTILS_H
