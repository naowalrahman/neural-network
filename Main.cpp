#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include "Layer.hpp"
#include "NeuralNetwork.hpp"
#include "Functions.hpp"

void print_model(NeuralNetwork& model) { 
    for (size_t i = 0; i < model.layers.size(); i++) {
        Layer& layer = model.layers[i];
        std::cout << "Layer " << i + 1 << " (" << layer.activation << "):\n\t";
        for (size_t j = 0; j < layer.weights.rows; j++) {
            std::cout << "Neuron " << j + 1 << ": ";
            for (size_t k = 0; k < layer.weights.cols; k++) {
                std::cout << layer.weights[j][k] << " ";
            }
            std::cout << "[bias: " << layer.biases[j][0] << "]";
            std::cout << "\n" << (j == layer.weights.rows - 1 ? "" : "\t");
        }
    }
}

void save_model(NeuralNetwork& model, std::string filename) { 
    std::ofstream file(filename);
    for (size_t i = 0; i < model.layers.size(); i++) {
        Layer& layer = model.layers[i];
        file << "Layer " << i + 1 << " (" << layer.activation << "):\n\t";
        for (size_t j = 0; j < layer.weights.rows; j++) {
            file << "Neuron " << j + 1 << ": ";
            for (size_t k = 0; k < layer.weights.cols; k++) {
                file << layer.weights[j][k] << " ";
            }
            file << "[bias: " << layer.biases[j][0] << "]";
            file << "\n" << (j == layer.weights.rows - 1 ? "" : "\t");
        }
    }
    file.close();
}

void test_feedforward() {
    Matrix input(2, 1);
    input[0][0] = 0.3;
    input[1][0] = 0.4;

    std::vector<Layer> layers = {
        Layer(4, "sigmoid"),
        Layer(3, "relu"),
        Layer(5, "sigmoid")
    };

    NeuralNetwork model(2, layers, "mean_squared_error");

    print_model(model);

    Matrix output = model.feedforward(input);
    std::cout << "Neural network output: ";
    for (size_t i = 0; i < output.rows; i++) {
        std::cout << output[i][0] << " ";
    }
    std::cout << "\n";

    output = model.feedforward(input);
    std::cout << "Neural network output: ";
    for (size_t i = 0; i < output.rows; i++) {
        std::cout << output[i][0] << " ";
    }
    std::cout << '\n';
}

void test_xor() {
    srand(time(0));

    std::vector<Layer> layers = {
        Layer(4, "sigmoid"),
        Layer(4, "sigmoid"),
        Layer(1, "sigmoid")
    };

    NeuralNetwork model(2, layers, "mean_squared_error");

    size_t n = 1000;
    std::vector<Matrix> train_inputs(n, Matrix(2, 1)), train_targets(n, Matrix(1, 1));
    std::vector<Matrix> test_inputs(n, Matrix(2, 1)), test_targets(n, Matrix(1, 1));

    for (size_t i = 0; i < n; i++) {
        train_inputs[i][0][0] = rand() % 2;
        train_inputs[i][1][0] = rand() % 2;
        train_targets[i][0][0] = train_inputs[i][0][0] != train_inputs[i][1][0];
    }

    for (size_t i = 0; i < n; i++) {
        test_inputs[i][0][0] = rand() % 2;
        test_inputs[i][1][0] = rand() % 2;
        test_targets[i][0][0] = test_inputs[i][0][0] != test_inputs[i][1][0];
    }

    model.train(train_inputs, train_targets, 0.3, 30, true);

    int correct = 0;
    double loss = 0;
    for (size_t i = 0; i < n; i++) {
        Matrix output = model.feedforward(test_inputs[i]);
        loss += Loss::loss_map.at(model.loss_function)(output, test_targets[i])[0][0];
        if ((output[0][0] > 0.5) == (test_targets[i][0][0] > 0.5)) {
            correct++;
        }
    }
    loss /= n;

    print_model(model);
    printf("Accuracy: %.5f, loss: %.8f\n", (double)correct / n, loss);
}

void read_mnist(std::string image_file, std::string label_file, std::vector<Matrix>& inputs, std::vector<Matrix>& targets, size_t num_images) {
    std::ifstream image_stream(image_file, std::ios::binary);
    std::ifstream label_stream(label_file, std::ios::binary);

    if (!image_stream.is_open() || !label_stream.is_open()) {
        throw std::runtime_error("Failed to open MNIST data files.");
    }

    int magic_number = 0;
    int num_items = 0;
    int rows = 0;
    int cols = 0;

    // Read image file header
    image_stream.read(reinterpret_cast<char*>(&magic_number), 4);
    image_stream.read(reinterpret_cast<char*>(&num_items), 4);
    image_stream.read(reinterpret_cast<char*>(&rows), 4);
    image_stream.read(reinterpret_cast<char*>(&cols), 4);

    magic_number = __builtin_bswap32(magic_number);
    num_items = __builtin_bswap32(num_items);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    // Read label file header
    label_stream.read(reinterpret_cast<char*>(&magic_number), 4);
    label_stream.read(reinterpret_cast<char*>(&num_items), 4);

    magic_number = __builtin_bswap32(magic_number);
    num_items = __builtin_bswap32(num_items);

    inputs.resize(num_images, Matrix(rows * cols, 1));
    targets.resize(num_images, Matrix(10, 1));

    for (size_t i = 0; i < num_images; ++i) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                unsigned char pixel = 0;
                image_stream.read(reinterpret_cast<char*>(&pixel), 1);
                inputs[i][r * cols + c][0] = pixel / 255.0;
            }
        }

        unsigned char label = 0;
        label_stream.read(reinterpret_cast<char*>(&label), 1);
        targets[i][label][0] = 1.0;
    }
}

void load_mnist_data(std::vector<Matrix>& train_inputs, std::vector<Matrix>& train_targets, std::vector<Matrix>& test_inputs, std::vector<Matrix>& test_targets) {
    read_mnist("../train-images-idx3-ubyte", "../train-labels-idx1-ubyte", train_inputs, train_targets, 60000);
    read_mnist("../t10k-images-idx3-ubyte", "../t10k-labels-idx1-ubyte", test_inputs, test_targets, 10000);
}

void test_mnist() {
    std::vector<Layer> layers = {
        Layer(128, "sigmoid"),
        Layer(10, "sigmoid")
    };

    // the input is a flattened 28x28 image (784 pixels)
    NeuralNetwork model(784, layers, "mean_squared_error");

    std::vector<Matrix> train_inputs, train_targets, test_inputs, test_targets;
    load_mnist_data(train_inputs, train_targets, test_inputs, test_targets);

    model.train(train_inputs, train_targets, 0.1, 10);

    int correct = 0;
    double loss = 0;
    for (size_t i = 0; i < test_inputs.size(); i++) {
        Matrix output = model.feedforward(test_inputs[i]);
        Matrix l = Loss::loss_map.at(model.loss_function)(output, test_targets[i]);
        for (size_t j = 0; j < l.rows; ++j) {
            loss += l[j][0] / l.rows;
        }

        if (output.max_index() == test_targets[i].max_index()) {
            correct++;
        }
    }
    loss /= test_inputs.size();
    std::cout << test_inputs.size() << "\n";

    save_model(model, "../mnist.txt");
    printf("Accuracy: %.5f, loss: %.8f\n", (double)correct / test_inputs.size(), loss);

    int ex_idx = 21;
    std::cout << "\nExample:\n";
    Matrix output = model.feedforward(test_inputs[ex_idx]);
    for (size_t i = 0; i < 28; i++) {
        for (size_t j = 0; j < 28; j++) {
            std::cout << (int)(test_inputs[ex_idx][28 * i + j][0] > 0.0);
        }
        std::cout << '\n';
    }
    for (size_t i = 0; i < 10; i++) {
        std::cout << output[i][0] << ' ';
    }
    std::cout << '\n';
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    std::cout.precision(5);
    // test_feedforward();
    // test_xor();
    test_mnist();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Total runtime: " << elapsed.count() << " seconds\n";
}
