#include "Layer.hpp"
#include <cmath>
#include <random>
#include <vector>
#include "Functions.hpp"

Layer::Layer(int n, std::string activation) {
    this->n = n;
    this->activation = activation;
}

void Layer::initialize_parameters(int fan_in) {
    int fan_out = this->n;
    double limit = std::sqrt(6.0 / (fan_in + fan_out));

    this->biases = Matrix(fan_out, 1);
    this->weights = Matrix(fan_out, fan_in);

    std::uniform_real_distribution<double> distribution(-limit, limit);
    std::random_device rd;
    std::default_random_engine engine(rd());
    auto get_weight = std::bind(distribution, engine);

    for (int i = 0; i < fan_out; i++) {
        for (int j = 0; j < fan_in; j++) {
            this->weights[i][j] = get_weight();
        }
        this->biases[i][0] = 0.0;
    }
}

Matrix Layer::calculate(Matrix& input) {
    Matrix z = this->weights.multiply(input).add(this->biases);
    auto activation_func = Activations::activation_map.at(activation);
    return activation_func(z);
}
