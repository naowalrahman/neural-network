#ifndef INCLUDE_NEURAL_NETWORK_FUNCTIONS_HPP_
#define INCLUDE_NEURAL_NETWORK_FUNCTIONS_HPP_

#include <functional>
#include <map>
#include <string>

namespace Activations {

double sigmoid(double x);
double relu(double x);
// TODO: Add more activation functions

const std::map<std::string, std::function<double(double)>> activation_map = {
    {"sigmoid", sigmoid},
    {"relu", relu},
};

}  // namespace Activations

#endif  // INCLUDE_NEURAL_NETWORK_FUNCTIONS_HPP_
