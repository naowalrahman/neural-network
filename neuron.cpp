#include <functional>
#include <vector>
#include "./linalg.cpp"

class Neuron {
public:
    std::vector<double> weights;
    double bias;
    std::function<double(double)> activation;

    Neuron(std::vector<double> weights,
           double bias,
           std::function<double(double)> activation) {
        this->weights = weights;
        this->bias = bias;
        this->activation = activation;
    }
    
    double feedforward(std::vector<double> inputs) {
        return activation(linalg::dot(inputs, weights) + bias) ;
    }
};
