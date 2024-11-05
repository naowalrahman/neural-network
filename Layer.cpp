#include "Layer.hpp"
#include <cmath>
#include <random>
#include <vector>
#include "Activations.hpp"
#include "Neuron.hpp"

Layer::Layer(int n, std::string activation) {
    this->n = n;
    this->activation = activation;
}

void Layer::initialize_parameters(int fan_in) {
    int fan_out = n;
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    
    std::uniform_real_distribution<double> distribution(-limit, limit);
    std::random_device rd;
    std::default_random_engine engine(rd());
    auto get_weight = std::bind(distribution, engine);

    for (int i = 0; i < fan_out; i++) {
        std::vector<double> weights(fan_in);
        for (int j = 0; j < fan_in; j++) {
            weights[j] = get_weight();
        }
        this->neurons.push_back(Neuron(weights, 0));
    }
}

std::vector<double> Layer::calculate(std::vector<double>& input) {
    std::vector<double> output(n);
    auto activation_func = Activations::activation_map.at(activation);

    for (int i = 0; i < n; i++) {
        output[i] = activation_func(this->neurons[i].calculate(input));
    }

    return output;
}
