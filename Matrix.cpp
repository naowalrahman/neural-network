#include "Matrix.hpp"
#include <stdexcept>

Matrix::Matrix() {
}

Matrix::Matrix(size_t rows, size_t cols, double val) {
    this->rows = rows;
    this->cols = cols;
    matrix = std::vector<std::vector<double>>(rows, std::vector<double>(cols, val));
}

Matrix::Matrix(size_t rows, size_t cols) : Matrix(rows, cols, 0.0) {}

std::vector<double>& Matrix::operator[](int i) {
    return matrix[i];
}

Matrix Matrix::multiply(Matrix& m) {
    if (cols != m.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    Matrix result(rows, m.cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            result[i][j] = 0;
            for (size_t k = 0; k < cols; ++k) {
                result[i][j] += matrix[i][k] * m[k][j];
            }
        }
    }

    return result;
}

Matrix Matrix::transpose() {
    Matrix result(cols, rows);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

Matrix Matrix::add(Matrix& m) {
    if (rows != m.rows || cols != m.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for addition.");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = matrix[i][j] + m[i][j];
        }
    }

    return result;
}

Matrix Matrix::subtract(Matrix& m) {
    if (rows != m.rows || cols != m.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = matrix[i][j] - m[i][j];
        }
    }

    return result;
}

Matrix Matrix::hadamard(Matrix& m) {
    if (rows != m.rows || cols != m.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for element-wise multiplication.");
    }

    Matrix result(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = matrix[i][j] * m[i][j];
        }
    }

    return result;
}

Matrix Matrix::scale(double scalar) {
    return this->apply_function([scalar](double val) { return val * scalar; });
}

Matrix Matrix::add_scalar(double scalar) {
    return this->apply_function([scalar](double val) { return val + scalar; });
}

Matrix Matrix::apply_function(std::function<double(double)> f) {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = f(matrix[i][j]);
        }
    }
    return result;
}

double Matrix::max_index() {
    double max = matrix[0][0];
    double index = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (matrix[i][j] > max) {
                max = matrix[i][j];
                index = i;
            }
        }
    }
    return index;
}