#ifndef LSTM_H
#define LSTM_H

#include <eigen3/Eigen/Dense>
#include <string>

class LSTM {
public:
    LSTM(int input_dim, int hidden_dim);
    void load_weights(const std::string &prefix);
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

private:
    int input_dim;
    int hidden_dim;
    Eigen::MatrixXd Wf, Wi, Wc, Wo;
    Eigen::MatrixXd Uf, Ui, Uc, Uo;
    Eigen::VectorXd bif, bii, bic, bio;
    Eigen::VectorXd bhf, bhi, bhc, bho;

    Eigen::VectorXd concatenate(const Eigen::VectorXd &a, const Eigen::VectorXd &b);
};

#endif // LSTM_H
