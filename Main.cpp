#include <iostream>
#include <string>
#include <vector>
#include "Layer.hpp"
#include "NeuralNetwork.hpp"

int main() {
    std::vector<double> input = {0.3, 0.4};
    std::string activation = "sigmoid";
    std::vector<Layer> layers = {
        Layer(2, activation)
    };

    NeuralNetwork model = NeuralNetwork(input, layers);
    
    for (size_t i = 0; i < model.layers.size(); i++) {
        std::cout << "Layer " << i << ":\n\t";
        std::vector<Neuron> neurons = model.layers[i].neurons;
        for (size_t j = 0; j < neurons.size(); j++) {
            std::cout << "Neuron " << j << ": ";
            for (size_t k = 0; k < neurons[j].weights.size(); k++) {
                std::cout << neurons[j].weights[k] << " ";
            }
            std::cout << "\n" << (j == neurons.size() - 1 ? "" : "\t");
        }
    }

    std::vector<double> output = model.feedforward();
    std::cout << "Neural network output: ";
    for(size_t i = 0; i < output.size(); i++) {
        std::cout << output[i] << " ";
    }
    std::cout << "\n";
}
