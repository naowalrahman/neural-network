#ifndef INCLUDE_NEURAL_NETWORK_LAYER_HPP_
#define INCLUDE_NEURAL_NETWORK_LAYER_HPP_

#include <string>
#include <vector>
#include "Matrix.hpp"

class Layer {
public:
    int n;
    std::string activation;
    Matrix weights;
    Matrix biases;

    Layer(int n, std::string activation);

    void initialize_parameters(int fan_in);
    Matrix calculate(Matrix& input);
};

#endif  // INCLUDE_NEURAL_NETWORK_LAYER_HPP_