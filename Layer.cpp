#include "Layer.hpp"
#include <vector>
#include "Functions.hpp"
#include "Neuron.hpp"

Layer::Layer(int n,
             std::string& activation,
             std::string& weight_initializer,
             std::string& bias_initializer) {
    this->n = n;
    this->activation = activation;
    this->weight_initializer = weight_initializer;
    this->bias_initializer = bias_initializer;

    this->neurons = std::vector<Neuron>(n);
    for (int i = 0; i < n; i++) {
        std::vector<double>& weights = Functions::weight_initializer_map.at(weight_initializer)();
        double bias = Functions::bias_initializer_map.at(bias_initializer)();
        neurons[i] = Neuron(weights, bias);
    }
}

std::vector<double>& Layer::calculate(std::vector<double>& input) {
    std::vector<double>& output = *(new std::vector<double>(n));
    auto activation_func = Functions::activation_map.at(activation);

    for (int i = 0; i < n; i++) {
        output[i] = activation_func(neurons[i].calculate(input));
    }

    return output;
}
