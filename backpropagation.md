# Understanding Backpropagation

Somewhat unintuitively, backpropagation has 2 steps: forward propagation and backward propagation. We first propagate the input forward through the network to get the activations at each layer. Then, we calculate the error at the output layer and propagate that error "backward" through the layers to calculate gradients for the loss function. I'll explain what all of this means below. 

## Forward Propagation
Consider all of the inputs $\mathbf{X}$ as a $n \times 1$ matrix, where n is the number of inputs. Then, for each hidden layer from $1 \dots L$ we have a weight matrix $\mathbf{W}^l$ and a bias vector $\mathbf{b}^l$. Each value $\mathbf{W}^l_{ij}$ represents the weight of the connection between the $i$-th neuron in layer $l$ and the the $j$-th layer in later $l - 1$. Similarly, each value $\mathbf{b}^l_i$ represents the bias term of the $i$-th neuron. All in all, we can represent each activation $\mathbf{a}^l$ recursively like this: 
$$ z^l = \mathbf{W}^l \mathbf{a}^{l-1} + \mathbf{b}^l $$
$$ \mathbf{a}^l = \sigma(\mathbf{z}^l) $$

where $\sigma$ is the activation function for the layer $l$. Note that $\mathbf{a}^0 = \mathbf{X}$.

By starting from the input layer and propagating to layer $L$, we arrive at $\mathbf{a}^L = \sigma(\mathbf{z}^l)$.

## Backward Propagation
Now that we've found the output, we should calculate the loss (or cost), measuring the difference between the predicted output with our current weights and biases and the expected output. We measure this with a loss function $\mathcal{L}$:
$$\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{n}\sum_{i=1}^{m}{\mathcal{L}(\mathbf{y}_i,\mathbf{\hat{y}}_i)}$$

Now, this is a good time to reinstate our goals: we want to find the gradient of the loss function $\partial \mathcal{L} / \partial \mathcal{\mathbf{W}^l}$ for each layer $l$ so that we can nudge the current values in a way that decreases the error. This is because the gradient always points in the [direction of greatest change](https://activecalculus.org/multi/S-10-6-Directional-Derivative.html), so by moving in the opposite direction of the gradient of $\mathcal{L}$ we slowly minimize it. 
