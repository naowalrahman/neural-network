#include "NeuralNetwork.hpp"
#include <vector>

NeuralNetwork::NeuralNetwork(std::vector<double>& input, std::vector<Layer>& layers) {
    this->input = input;
    this->layers = layers;
}

std::vector<double>& NeuralNetwork::feedforward() {
    std::vector<double>& current_input = this->input;
    for(int i = 0; i < this->layers.size(); i++) {
        current_input = layers[i].calculate(current_input);  
    }
    return current_input;
}
