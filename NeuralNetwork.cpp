#include "NeuralNetwork.hpp"
#include <vector>

NeuralNetwork::NeuralNetwork(std::vector<double>& input,
                             std::vector<Layer>& layers) {
    this->input = input;
    this->layers = layers;

    for (size_t i = 0; i < this->layers.size(); i++) {
        int fan_in = i == 0 ? input.size() : this->layers[i - 1].n;
        this->layers[i].initialize_parameters(fan_in);
    }
}

std::vector<double>& NeuralNetwork::feedforward() {
    std::vector<double>& current_input = this->input;
    for (size_t i = 0; i < this->layers.size(); i++) {
        current_input = this->layers[i].calculate(current_input);
    }
    return current_input;
}
