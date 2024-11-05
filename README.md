# Neural Network

**WIP.**

This is an attempt at implementing a neural network in purely C++ with no external libraries (only C++ STL). So far, I've completed the entire neural network structure, including [glorot uniform weight initialization](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform) and multiple activation function support. I'm currently working on implementing backpropagation to train the neural network.

I will document this project more extensively and give this README a more formal writeup once I finish implementation!

For now, to build/run:

```bash
git clone https://github.com/naowalrahman/neural-network.git
cd neural-network
make
./Main
```

Edit `Main.cpp` to change the setup of the neural network :relaxed:.
