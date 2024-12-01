#ifndef INCLUDE_NEURAL_NETWORK_FUNCTIONS_HPP_
#define INCLUDE_NEURAL_NETWORK_FUNCTIONS_HPP_

#include <functional>
#include <map>
#include <string>
#include "Matrix.hpp"

namespace Activations {

    Matrix sigmoid(Matrix& x);
    Matrix relu(Matrix& x);
    Matrix identity(Matrix& x);
    Matrix sigmoid_deriv(Matrix& x);
    Matrix relu_deriv(Matrix& x);
    Matrix identity_deriv(Matrix& x);

    const std::unordered_map<std::string, std::function<Matrix(Matrix&)>> activation_map = {
        {"sigmoid", sigmoid},
        {"relu", relu},
        {"identity", identity},
    };

    const std::unordered_map<std::string, std::function<Matrix(Matrix&)>> activation_deriv_map = {
        {"sigmoid", sigmoid_deriv},
        {"relu", relu_deriv},
        {"identity", identity_deriv},
    };

}  // namespace Activations

namespace Loss {

    Matrix mean_squared_error(Matrix& predicted, Matrix& target);
    Matrix mean_squared_error_deriv(Matrix& predicted, Matrix& target);
    // TODO: Add more activation functions

    const std::unordered_map<std::string, std::function<Matrix(Matrix&, Matrix&)>> loss_map = {
        {"mean_squared_error", mean_squared_error},
    };

    const std::unordered_map<std::string, std::function<Matrix(Matrix&, Matrix&)>> loss_deriv_map = {
        {"mean_squared_error", mean_squared_error_deriv},
    };

} // namespace Loss

#endif  // INCLUDE_NEURAL_NETWORK_FUNCTIONS_HPP_