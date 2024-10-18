#ifndef INCLUDE_NEURAL_NETWORK_LAYER_HPP_
#define INCLUDE_NEURAL_NETWORK_LAYER_HPP_

#include <string>
#include <vector>
#include "Neuron.hpp"

class Layer {
public:
    int n;
    std::string activation, weight_initializer, bias_initializer;
    std::vector<Neuron> neurons;

    Layer(int n,
          std::string& activation,
          std::string& weight_initializer,
          std::string& bias_initializer);

    std::vector<double>& calculate(std::vector<double>& inputs);
};

#endif  // INCLUDE_NEURAL_NETWORK_LAYER_HPP_
