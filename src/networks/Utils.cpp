#include "networks/Utils.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
Eigen::MatrixXd readCSV(const std::string &path) {
    std::ifstream file(path);
    std::string line, cell;
    std::vector<std::vector<double>> data;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::vector<double> row;
        while (std::getline(lineStream, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    
    }
    Eigen::MatrixXd mat(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[i].size(); ++j) {
            mat(i, j) = data[i][j];
        }
    }

    return mat;
}

Eigen::MatrixXd relu(const Eigen::MatrixXd &x) {
    return x.array().max(0);
}

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &x) {
    return 1.0 / (1.0 + (-x.array()).exp());
}

Eigen::MatrixXd tanh(const Eigen::MatrixXd &x) {
    return x.array().tanh();
}
