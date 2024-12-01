#ifndef INCLUDE_NEURAL_NETWORK_MATRIX_HPP_
#define INCLUDE_NEURAL_NETWORK_MATRIX_HPP_

#include <cstdlib>
#include <vector>
#include <functional>

struct Matrix {
    std::vector<std::vector<double>> matrix;
    size_t rows;
    size_t cols;

    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double val);

    std::vector<double>& operator[](int i);
    Matrix multiply(Matrix& m);
    Matrix transpose();
    Matrix add(Matrix& m);
    Matrix subtract(Matrix& m);
    Matrix hadamard(Matrix& m);
    Matrix scale(double scalar);
    Matrix add_scalar(double scalar);
    Matrix apply_function(std::function<double(double)>f);
    double max_index();
};

#endif // INCLUDE_NEURAL_NETWORK_MATRIX_HPP_