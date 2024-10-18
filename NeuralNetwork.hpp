#ifndef INCLUDE_NEURAL_NETWORK_NEURALNETWORK_HPP_
#define INCLUDE_NEURAL_NETWORK_NEURALNETWORK_HPP_

#include <vector>
#include "Layer.hpp"

class NeuralNetwork {
public:
    std::vector<double> input;
    std::vector<Layer> layers; // last layer must be output layer

    NeuralNetwork(std::vector<double>& input, std::vector<Layer>& layers);
    
    std::vector<double>& feedforward();

};

#endif  // INCLUDE_NEURAL_NETWORK_NEURALNETWORK_HPP_
