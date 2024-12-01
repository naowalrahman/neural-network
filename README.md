# Neural Network

**WIP.**

This is an attempt at implementing a neural network in purely C++ with no external libraries (only C++ STL). So far, I've completed the entire neural network structure, which uses [glorot uniform weight initialization](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform) and has multiple activation function support. Training is done via [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) using [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). 

I will document this project more extensively and give this README a more formal write-up once I finish implementation! For an explanation of backpropagation, see [backpropagation.md](./backpropagation.md).

## XOR Performance

The following model achieves essentially 100% accuracy after training with $n = 1000$ random XOR samples, learning rate $\eta = 0.3$, and 30 epochs.

```txt
Layer 1 (sigmoid):
        Neuron 1: -3.9019 1.5016 
        Neuron 2: 5.2515 5.5733 
        Neuron 3: -1.6861 4.3022 
        Neuron 4: -2.5383 0.79651 
Layer 2 (sigmoid):
        Neuron 1: -2.6143 -2.7931 3.1181 -1.0519 
        Neuron 2: 3.0904 4.9213 -4.5853 2.0534 
        Neuron 3: 0.5383 0.72858 0.55805 0.01614 
        Neuron 4: 0.61733 1.0527 0.40602 -0.091099 
Layer 3 (sigmoid):
        Neuron 1: -5.2896 8.2978 -0.0018658 0.18326 
Accuracy: 1.00000, loss: 0.00015695
```

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
- [ ] MNIST classification model
