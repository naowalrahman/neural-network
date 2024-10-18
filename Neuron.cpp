#include "Neuron.hpp"
#include "Linalg.hpp"

Neuron::Neuron(std::vector<double>& weights, double bias) {
    this->weights = weights;
    this->bias = bias;
}

double Neuron::calculate(std::vector<double>& inputs) {
    return Linalg::dot(inputs, weights) + bias;
}
