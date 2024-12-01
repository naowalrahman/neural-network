#include "NeuralNetwork.hpp"
#include <vector>
#include "Functions.hpp"
#include <iostream>

NeuralNetwork::NeuralNetwork(double input_size, std::vector<Layer>& layers, std::string loss_function) {
    this->input_size = input_size;
    this->layers = layers;
    this->loss_function = loss_function;

    for (size_t i = 0; i < this->layers.size(); i++) {
        int fan_in = i == 0 ? input_size : this->layers[i - 1].n;
        this->layers[i].initialize_parameters(fan_in);
    }
}

Matrix NeuralNetwork::feedforward(Matrix& input) {
    Matrix current_input = input;
    for (size_t i = 0; i < this->layers.size(); i++) {
        current_input = this->layers[i].calculate(current_input);
    }
    return current_input;
}

void NeuralNetwork::backpropagate(Matrix& input, Matrix& target, double learning_rate) {
    std::vector<Matrix> outputs(this->layers.size()), activated_outputs(this->layers.size());
    auto activation_map = Activations::activation_map;
    auto activation_deriv_map = Activations::activation_deriv_map;

    outputs[0] = this->layers[0].weights.multiply(input).add(this->layers[0].biases);
    activated_outputs[0] = activation_map.at(this->layers[0].activation)(outputs[0]);
    for (size_t i = 1; i < this->layers.size(); i++) {
        outputs[i] = this->layers[i].weights.multiply(activated_outputs[i - 1]).add(this->layers[i].biases);
        activated_outputs[i] = activation_map.at(this->layers[i].activation)(outputs[i]);
    }

    // Compute loss derivative with respect to the last layer's activated outputs
    Matrix loss_deriv = Loss::loss_deriv_map.at(loss_function)(activated_outputs.back(), target);

    // Propagate the error backward
    for (int i = this->layers.size() - 1; i >= 0; --i) {
        Matrix deriv = activation_deriv_map.at(this->layers[i].activation)(outputs[i]);
        Matrix delta = loss_deriv.hadamard(deriv);

        Matrix input_to_use = (i == 0) ? input : activated_outputs[i - 1];
        input_to_use = input_to_use.transpose();
        Matrix weight_gradient = delta.multiply(input_to_use);

        Matrix scaled_weight_gradient = weight_gradient.scale(learning_rate);
        Matrix scaled_delta = delta.scale(learning_rate);

        this->layers[i].weights = this->layers[i].weights.subtract(scaled_weight_gradient);
        this->layers[i].biases = this->layers[i].biases.subtract(scaled_delta);

        if (i != 0) {
            loss_deriv = this->layers[i].weights.transpose().multiply(delta);
        }
    }
}

void NeuralNetwork::train(std::vector<Matrix>& inputs, std::vector<Matrix>& targets, double learning_rate, int epochs, bool verbose) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (size_t i = 0; i < inputs.size(); i++) {
            this->backpropagate(inputs[i], targets[i], learning_rate);
            if (verbose) {
                Matrix output = this->feedforward(inputs[i]);
                Matrix loss = Loss::loss_map.at(loss_function)(output, targets[i]);
                std::cout << "Epoch " << epoch << ", loss: " << loss[0][0] << '\r';
                std::cout.flush();
            }
        }
    }
    if (verbose) std::cout << "\n";
}