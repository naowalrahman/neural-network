#ifndef INCLUDE_NEURAL_NETWORK_NEURALNETWORK_HPP_
#define INCLUDE_NEURAL_NETWORK_NEURALNETWORK_HPP_

#include <vector>
#include "Layer.hpp"
#include "Matrix.hpp"

class NeuralNetwork {
public:
    double input_size;
    std::vector<Layer> layers; // last layer must be output layer
    std::string loss_function;

    NeuralNetwork(double input, std::vector<Layer>& layers, std::string loss_function);

    Matrix feedforward(Matrix& input);
    void backpropagate(Matrix& input, Matrix& target, double learning_rate);
    void train(std::vector<Matrix>& inputs, std::vector<Matrix>& targets, double learning_rate, int epochs, bool verbose=true);
};

#endif  // INCLUDE_NEURAL_NETWORK_NEURALNETWORK_HPP_