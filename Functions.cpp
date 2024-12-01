#include "Functions.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "Matrix.hpp"

Matrix Activations::sigmoid(Matrix& x) {
    return x.apply_function([](double val) { return 1 / (1 + std::exp(-val)); });
}

Matrix Activations::relu(Matrix& x) {
    return x.apply_function([](double val) { return val > 0.0 ? val : 0.01 * val; });
}

Matrix Activations::identity(Matrix& x) {
    return x;
}

Matrix Activations::sigmoid_deriv(Matrix& x) {
    Matrix sig = sigmoid(x);
    Matrix one_minus_sig = sig.scale(-1.0).add_scalar(1.0);
    return sig.hadamard(one_minus_sig);
}

Matrix Activations::relu_deriv(Matrix& x) {
    return x.apply_function([](double val) { return val > 0.0 ? 1.0 : 0.01; });
}

Matrix Activations::identity_deriv(Matrix& x) {
    static Matrix m = Matrix(x.rows, x.cols, 1.0);
    return m;
} 

Matrix Loss::mean_squared_error(Matrix& predicted, Matrix& target) {
    if (predicted.rows != target.rows || predicted.cols != target.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for MSE calculation.");
    }

    Matrix diff = predicted.subtract(target);
    Matrix squared = diff.hadamard(diff);
    return squared.scale(0.5);
}

Matrix Loss::mean_squared_error_deriv(Matrix& predicted, Matrix& target) {
    if (predicted.rows != target.rows || predicted.cols != target.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for MSE derivative calculation.");
    }

    return predicted.subtract(target);
}