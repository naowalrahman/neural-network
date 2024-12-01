# Neural Network

This is an attempt at implementing a neural network in purely C++ with no external libraries (only C++ STL). So far, I've completed the entire neural network structure, which uses [glorot uniform weight initialization](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform) and has multiple activation function support. Training is done via [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) using [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). 

I will document this project more extensively and give this README a more formal write-up once I finish implementation! For an explanation of backpropagation, see [backpropagation.md](./backpropagation.md).

## XOR Performance

The following model achieves essentially 100% accuracy after training with $n = 1000$ random XOR samples, learning rate $\eta = 0.3$, and 30 epochs.

```txt
Layer 1 (sigmoid):
        Neuron 1: 1.7349 1.7117 [bias: -2.5654]
        Neuron 2: -1.9663 -2.4515 [bias: 3.3307]
        Neuron 3: -5.4436 -5.2983 [bias: 1.9498]
        Neuron 4: 1.4488 0.54997 [bias: -0.98019]
Layer 2 (sigmoid):
        Neuron 1: 0.15555 0.86921 -1.9651 0.67768 [bias: 0.26103]
        Neuron 2: 1.6101 -1.2021 2.1533 0.6044 [bias: -0.62403]
        Neuron 3: 2.334 -3.3788 4.4697 1.6803 [bias: -0.53567]
        Neuron 4: -2.0916 2.4694 -4.9442 -1.2314 [bias: 0.85497]
Layer 3 (sigmoid):
        Neuron 1: 1.8202 -3.0607 -6.5665 6.2184 [bias: 0.16765]
Accuracy: 1.00000, loss: 0.00012206
```

## MNIST Performance

The model available at [mnist.txt](./mnist.txt) achieves 97.49% accuracy on the [MNIST handwritten digits dataset](https://yann.lecun.com/exdb/mnist/). It was trained with the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) loss function using a learning rate $\eta = 0.1$ and 10 epochs.

## Building

Ensure `cmake` is installed on your system. For now, to build/run:

```bash
git clone https://github.com/naowalrahman/neural-network.git
cd neural-network
chmod +x build.sh
./build.sh Release
```

Edit `Main.cpp` to change the setup of the neural network :relaxed:.

## To Do

- [x] Feedforward ANN architecture
- [x] Backpropagation algorithm to train models
- [x] Train xor model
- [x] MNIST classification model
- [ ] Perspicuously document code
