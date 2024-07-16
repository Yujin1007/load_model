#include <eigen3/Eigen/Dense>
#include <iostream>
#include "networks/Actor.h"

using namespace std;
int main() {
    // Define dimensions
    int state_dim = 31;
    int lstm_hidden_dim = 256;
    int mlp_hidden_dim = 256;
    int output_dim = 4;

    cout<<" Create the Actor model"<<endl;
    Actor actor(state_dim, lstm_hidden_dim, mlp_hidden_dim, output_dim);
    cout<<" Load model parameters from CSV files"<<endl;
    std::string directory = "/home/kist-robot2/Franka/franka_valve/py_src/weight/rotation/";
    actor.load_weights(directory);
    
    cout<<" Create a sample input (batch size 1, sequence length 10, state_dim features)"<<endl;
    Eigen::MatrixXd input = Eigen::MatrixXd::Ones(state_dim, 10);

    cout<<" Forward pass"<<endl;
    Eigen::MatrixXd output = actor.forward(input);

    cout << "Model output: \n" << output << endl;

    return 0;
}
