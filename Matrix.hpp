#ifndef INCLUDE_NEURAL_NETWORK_MATRIX_HPP_
#define INCLUDE_NEURAL_NETWORK_MATRIX_HPP_

#include <cstdlib>
#include <vector>

struct Matrix {
    std::vector<std::vector<double>> matrix;
    size_t rows;
    size_t cols;

    Matrix();
    Matrix(size_t rows, size_t cols);

    std::vector<double>& operator[](int i);
    Matrix multiply(Matrix& m);
    Matrix add(Matrix& m);
    Matrix subtract(Matrix& m);
    Matrix hadamard(Matrix& m);
    Matrix scale(double scalar);
    Matrix add_scalar(double scalar);
    double argmax();
    Matrix transpose();
};

#endif // INCLUDE_NEURAL_NETWORK_MATRIX_HPP_