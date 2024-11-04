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
    
    for (int i = 0; i < layers.size(); i++) {
        std::cout << "Layer " << i << ":\n\t";
        for (int j = 0; j < layers[i].neurons.size(); j++) {
            std::cout << "Neuron " << j << ": ";
            for (int k = 0; k < layers[i].neurons[j].weights.size(); k++) {
                std::cout << layers[i].neurons[j].weights[k] << " ";
            }
            std::cout << "\n";
        }
    }

    NeuralNetwork model(input, layers);
}
