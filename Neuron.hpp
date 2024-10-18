#ifndef INCLUDE_NEURAL_NETWORK_NEURON_HPP_
#define INCLUDE_NEURAL_NETWORK_NEURON_HPP_

#include <vector>

class Neuron {
public:
    std::vector<double> weights;
    double bias;

    Neuron(std::vector<double>& weights, double bias);

    double calculate(std::vector<double>& inputs);
};

#endif  // INCLUDE_NEURAL_NETWORK_NEURON_HPP_
