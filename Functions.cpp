#include "Functions.hpp"
#include <cmath>
#include <stdexcept>
#include "Matrix.hpp"

Matrix Activations::sigmoid(Matrix& x) {
    Matrix result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; i++) {
        for (size_t j = 0; j < x.cols; j++) {
            result[i][j] = 1 / (1 + std::exp(-x[i][j]));
        }
    }
    return result;
}

Matrix Activations::relu(Matrix& x) {
    Matrix result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; i++) {
        for (size_t j = i; j < x.cols; j++) {
            result[i][j] = std::max(0.0, x[i][j]);
        }
    }
    return result;
}

Matrix Activations::sigmoid_derivative(Matrix& x) {
    Matrix sig = sigmoid(x);
    Matrix one_minus_sig = sig.scale(-1.0).add_scalar(1.0);
    return sig.hadamard(one_minus_sig);
}

Matrix Activations::relu_derivative(Matrix& x) {
    Matrix result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < x.cols; ++j) {
            result[i][j] = x[i][j] > 0.0 ? 1.0 : 0.0;
        }
    }
    return result;
}

Matrix Activations::identity(Matrix& x) {
    return x;
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