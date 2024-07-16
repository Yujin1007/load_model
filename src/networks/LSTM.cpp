#include "networks/LSTM.h"
#include "networks/Utils.h"

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream>

// Utility function to read CSV files into Eigen matrices
// Eigen::MatrixXd readCSV(const std::string &path) {
//     std::ifstream file(path);
//     std::string line, cell;
//     std::vector<std::vector<double>> data;

//     while (std::getline(file, line)) {
//         std::stringstream lineStream(line);
//         std::vector<double> row;
//         while (std::getline(lineStream, cell, ',')) {
//             row.push_back(std::stod(cell));
//         }
//         data.push_back(row);
//     }

//     Eigen::MatrixXd mat(data.size(), data[0].size());
//     for (int i = 0; i < data.size(); ++i) {
//         for (int j = 0; j < data[i].size(); ++j) {
//             mat(i, j) = data[i][j];
//         }
//     }

//     return mat;
// }

// Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &x) {
//     std::cout<<"sigmoid"<<std::endl;
//     return 1.0 / (1.0 + (-x.array()).exp());
// }

// Eigen::MatrixXd tanh(const Eigen::MatrixXd &x) {
//     return x.array().tanh();
// }

LSTM::LSTM(int input_dim, int hidden_dim) : input_dim(input_dim), hidden_dim(hidden_dim) {}

void LSTM::load_weights(const std::string &prefix)
{
    Wf = readCSV(prefix + "lstm_weight_ih_l0_forget.csv");
    Uf = readCSV(prefix + "lstm_weight_hh_l0_forget.csv");
    bif = readCSV(prefix + "lstm_bias_ih_l0_forget.csv");
    bhf = readCSV(prefix + "lstm_bias_hh_l0_forget.csv");

    Wi = readCSV(prefix + "lstm_weight_ih_l0_input.csv");
    Ui = readCSV(prefix + "lstm_weight_hh_l0_input.csv");
    bii = readCSV(prefix + "lstm_bias_ih_l0_input.csv");
    bhi = readCSV(prefix + "lstm_bias_hh_l0_input.csv");

    Wc = readCSV(prefix + "lstm_weight_ih_l0_cell.csv");
    Uc = readCSV(prefix + "lstm_weight_hh_l0_cell.csv");
    bic = readCSV(prefix + "lstm_bias_ih_l0_cell.csv");
    bhc = readCSV(prefix + "lstm_bias_hh_l0_cell.csv");

    Wo = readCSV(prefix + "lstm_weight_ih_l0_output.csv");
    Uo = readCSV(prefix + "lstm_weight_hh_l0_output.csv");
    bio = readCSV(prefix + "lstm_bias_ih_l0_output.csv");
    bho = readCSV(prefix + "lstm_bias_hh_l0_output.csv");
}

// Eigen::MatrixXd LSTM::forward(const Eigen::MatrixXd &input) {
//     int seq_len = input.cols();
//     Eigen::MatrixXd h = Eigen::MatrixXd::Zero(hidden_dim, seq_len);
//     Eigen::MatrixXd c = Eigen::MatrixXd::Zero(hidden_dim, seq_len);

//     Eigen::VectorXd h_prev = Eigen::VectorXd::Zero(hidden_dim);
//     Eigen::VectorXd c_prev = Eigen::VectorXd::Zero(hidden_dim);

//     for (int t = 0; t < seq_len; ++t) {

//         Eigen::VectorXd x_t = input.col(t);
//         Eigen::VectorXd concat = concatenate(x_t, h_prev);

//     std::cout << "Shape of Wf: (" << Wf.rows() << ", " << Wf.cols() << ")\n";
//     std::cout << "Shape of Uf: (" << Uf.rows() << ", " << Uf.cols() << ")\n";
//     std::cout << "Shape of x_t: (" << x_t.size() << ", 1)\n";
//     std::cout << "Shape of bf: (" << bf.size() << ", 1)\n";
//     std::cout << "Shape of h_prev: (" << h_prev.size() << ", 1)\n";

//         Eigen::VectorXd f_t = sigmoid(Wf * x_t + bf + Uf * h_prev);
//         Eigen::VectorXd i_t = sigmoid(Wi * x_t + bi + Ui * h_prev);
//         Eigen::VectorXd c_tilde_t = tanh(Wc * x_t + bc + Uc * h_prev);
//         Eigen::VectorXd o_t = sigmoid(Wo * x_t + bo + Uo * h_prev);

//         std::cout<<"5"<<std::endl;
//     std::cout << "Shape of f_t: (" << f_t.size() << ", 1)\n";
//     std::cout << "Shape of c_prev: (" << c_prev.size() << ", 1)\n";
//     std::cout << "Shape of c_tilde_t: (" << c_tilde_t.size() << ", 1)\n";
//     std::cout << "Shape of i_t: (" << i_t.size() << ", 1)\n";

//     std::cout << "Shape of i_t: (" << (f_t.cwiseProduct(c_prev)).size() << ", 1)\n";

//     std::cout << "Shape of i_t: (" << (i_t.cwiseProduct(c_tilde_t)).size() << ", 1)\n";
//         c_prev = f_t.cwiseProduct(c_prev) + i_t.cwiseProduct(c_tilde_t);
//         std::cout<<"6"<<std::endl;

//         h_prev = o_t.cwiseProduct(tanh(c_prev));
//     std::cout<<t<<std::endl;

//         h.col(t) = h_prev;
//         c.col(t) = c_prev;
//     }
//     return h;
// }

Eigen::MatrixXd LSTM::forward(const Eigen::MatrixXd &input)
{
    int seq_len = input.cols();
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(hidden_dim, seq_len);
    Eigen::MatrixXd c = Eigen::MatrixXd::Zero(hidden_dim, seq_len);

    Eigen::VectorXd h_prev = Eigen::VectorXd::Zero(hidden_dim);
    Eigen::VectorXd c_prev = Eigen::VectorXd::Zero(hidden_dim);

    for (int t = 0; t < seq_len; ++t)
    {
        Eigen::VectorXd x_t = input.col(t);
        // Eigen::VectorXd concat = concatenate(x_t, h_prev);
        Eigen::VectorXd f_t = sigmoid(Wf * x_t + bif + Uf * h_prev + bhf);
        Eigen::VectorXd i_t = sigmoid(Wi * x_t + bii + Ui * h_prev + bhi);
        Eigen::VectorXd c_tilde_t = tanh(Wc * x_t + bic + Uc * h_prev + bhc);
        Eigen::VectorXd o_t = sigmoid(Wo * x_t + bio + Uo * h_prev + bho);

        c_prev = f_t.cwiseProduct(c_prev) + i_t.cwiseProduct(c_tilde_t);
        h_prev = o_t.cwiseProduct(tanh(c_prev));

        h.col(t) = h_prev;
        c.col(t) = c_prev;
    }
    return h;
}

Eigen::VectorXd LSTM::concatenate(const Eigen::VectorXd &a, const Eigen::VectorXd &b)
{
    Eigen::VectorXd result(a.size() + b.size());
    result << a, b;
    return result;
}
