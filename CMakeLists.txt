cmake_minimum_required(VERSION 3.10)
project(load_model)

set(CMAKE_CXX_STANDARD 14)

# Add Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(include)

add_executable(actor_model src/main.cpp src/networks/LSTM.cpp src/networks/MLP.cpp src/networks/Linear.cpp src/networks/LayerNorm.cpp src/networks/NormalizedMLP.cpp src/networks/Networks.cpp src/networks/Utils.cpp)
target_link_libraries(actor_model Eigen3::Eigen)
