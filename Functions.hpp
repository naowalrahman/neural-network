#ifndef INCLUDE_NEURAL_NETWORK_FUNCTIONS_HPP_
#define INCLUDE_NEURAL_NETWORK_FUNCTIONS_HPP_

#include <functional>
#include <map>
#include <string>

namespace Functions {

namespace Activations {

double sigmoid(double x);
double relu(double x);
// TODO: Add more activation functions

}  // namespace Activations

namespace WeightInitializers {

std::vector<double>& glorot_uniform();
// TODO: Add more weight initializer functions

}  // namespace WeightInitializers

namespace BiasInitializers {

double zeros();
// TODO: Add more bias initializer functions

}  // namespace BiasInitializers

const std::map<std::string, std::function<double(double)>> activation_map = {
    {"sigmoid", Activations::sigmoid},
    {"relu", Activations::relu},
};

const std::map<std::string, std::function<std::vector<double>&(void)>> weight_initializer_map = {
    {"glorot_uniform", WeightInitializers::glorot_uniform}
};

const std::map<std::string, std::function<double(void)>> bias_initializer_map = {
    {"zeros", BiasInitializers::zeros}
};

}  // namespace Functions

#endif  // INCLUDE_NEURAL_NETWORK_FUNCTIONS_HPP_
