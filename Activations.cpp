#include "Activations.hpp"
#include <cmath>

double Activations::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double Activations::relu(double x) {
    return std::max(0.0, x); 
}
