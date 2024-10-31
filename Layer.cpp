#include "Layer.hpp"
#include <cmath>
#include <random>
#include <vector>
#include "Activations.hpp"
#include "Neuron.hpp"

Layer::Layer(int n, std::string& activation) {
    this->n = n;
    this->activation = activation;

    this->neurons = std::vector<Neuron>(n);
    for (int i = 0; i < n; i++) {
        std::vector<double> weights(n);
        neurons[i] = Neuron(weights, 0);
    }
}

void Layer::initialize_parameters(int fan_in, int fan_out) {
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    std::uniform_real_distribution<double> distribution(-limit, limit);
    std::default_random_engine engine;
    for (Neuron& neuron : neurons) {
        for (int j = 0; j < neuron.weights.size(); j++) {
            neuron.weights[j] = distribution(engine);
        }
    }
}

std::vector<double>& Layer::calculate(std::vector<double>& input) {
    std::vector<double>& output = *(new std::vector<double>(n));
    auto activation_func = Activations::activation_map.at(activation);

    for (int i = 0; i < n; i++) {
        output[i] = activation_func(neurons[i].calculate(input));
    }

    return output;
}
