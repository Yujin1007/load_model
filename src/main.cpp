#include <eigen3/Eigen/Dense>
#include <iostream>
#include "networks/Networks.h"
#include <chrono>

using namespace std;
int main() {
    // Define dimensions
    int state_dim = 31;
    int lstm_hidden_dim = 256;
    array<int,2> mlp_hidden_dim = {256,256};
    // array<int,2> mlp_hidden_dim = {512,256};
    int output_dim1 = 4;
    int output_dim2 = 2;

    Classifier classifier(8, 20);

    // cout<<" Create the Actor model"<<endl;
    // Actor actorr(state_dim, mlp_hidden_dim, output_dim1);
    // Actor actorf(state_dim, mlp_hidden_dim, output_dim2);
    // cout<<" Load model parameters from CSV files"<<endl;
    // std::string directory1 = "/home/kist-robot2/Franka/franka_valve_new/py_src/weight/rotation/";
    // actorr.load_weights(directory1);
    // std::string directory2 = "/home/kist-robot2/Franka/franka_valve_new/py_src/weight/force/";
    // actorf.load_weights(directory2);

    // cout<<" Create a sample input (batch size 1, sequence length 10, state_dim features)"<<endl;
    // Eigen::MatrixXd input = Eigen::MatrixXd::Ones(state_dim, 1);
    ActorR actorr(state_dim, lstm_hidden_dim, mlp_hidden_dim, output_dim1);
    ActorF actorf(state_dim, lstm_hidden_dim, mlp_hidden_dim, output_dim2);
    cout<<" Load model parameters from CSV files"<<endl;
    std::string directory1 = "/home/kist-robot2/Desktop/RA-L/load_model/src/networks/weights/rotation/";
    actorr.load_weights(directory1);
    std::string directory2 = "/home/kist-robot2/Desktop/RA-L/load_model/src/networks/weights/force/";
    actorf.load_weights(directory2);

    cout<<" Create a sample input (batch size 1, sequence length 10, state_dim features)"<<endl;
    Eigen::MatrixXd input = Eigen::MatrixXd::Ones(state_dim, 10);


    cout<<" Forward pass"<<endl;
//    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    // Perform the calculation
    Eigen::MatrixXd output1 = actorr.forward(input);
    Eigen::MatrixXd output2 = actorf.forward(input);

    // Get the ending timepoint
    std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
    std::cout << "for문을 돌리는데 걸리는 시간(초) : " << sec.count() *1000<<"ms"<< std::endl;
    // Calculate the duration in milliseconds
//    std::chrono::duration<double, std::milli> duration = end - start;

    // Print the mean value and the duration


//    std::cout << "Calculation time: " << duration.count() << " ms" << std::endl;
//    cout << "Model output: \n" << output << endl;

    return 0;
}
