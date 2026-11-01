# SNN: a neural network from scratch in C++

A small feed-forward neural network library written in C++17 with **no external dependencies**: no Eigen, no BLAS, just the standard library. Every piece is written by hand: the matrix utilities, the layers, backpropagation, the loss and the training loop.

It was built as a learning project to understand how neural networks work internally, not how to use a framework. It learns XOR in a few hundred epochs.

## Features

- Modular design: every layer implements a common `Layer` interface (`forward` / `backward`)
- `Linear` (fully connected) layer with weight and bias updates by gradient descent
- A generic `Activation` layer: pass any function and its derivative (sigmoid and ReLU included, lambdas work too)
- Mean squared error loss and its derivative
- Per-sample stochastic gradient descent (SGD) training loop
- Hand-written linear algebra helpers: dot, outer product, transpose, matrix-vector product, Hadamard product and more
- Unit tests with values computed by hand

## Project structure

```
.
├── includes/
│   ├── snn.hpp               # Vec / Mat types, math, activation and loss declarations
│   ├── layers.hpp            # Layer (interface), Linear, Activation
│   └── neural_network.hpp    # NeuralNetwork class
├── src/
│   ├── math.cpp              # linear algebra utilities, random initialisation
│   ├── activation.cpp        # sigmoid, ReLU, Activation layer
│   ├── linear.cpp            # Linear layer
│   ├── loss.cpp              # mse, mseDerivative
│   ├── neural_network.cpp    # add / predict / fit
│   └── main.cpp              # XOR example
└── tests/
    └── test_snn.cpp          # unit tests
```

## Build and run

Requires a C++17 compiler (tested with g++).

```bash
# build the XOR example
g++ -O2 -std=c++17 -Wall -Wextra -Iincludes \
    src/main.cpp src/math.cpp src/activation.cpp src/linear.cpp src/loss.cpp src/neural_network.cpp \
    -o snn

./snn
```

Run the unit tests:

```bash
g++ -std=c++17 -Wall -Wextra -Iincludes \
    tests/test_snn.cpp src/math.cpp src/activation.cpp src/linear.cpp src/loss.cpp \
    -o test

./test
```

Each check prints `[PASS]` or `[FAIL]`, and the program's exit code is the number of failures.

## Example: learning XOR

```cpp
std::vector<Vec> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
std::vector<Vec> Y = {{0}, {1}, {1}, {0}};

NeuralNetwork net;
net.add(std::make_unique<Linear>(2, 3));
net.add(std::make_unique<Activation>(sigmoid, sigmoidDerivative));
net.add(std::make_unique<Linear>(3, 1));
net.add(std::make_unique<Activation>(sigmoid, sigmoidDerivative));

net.fit(X, Y, 10000, 0.5);   // data, epochs, learning rate

for (size_t i = 0; i < X.size(); ++i)
    std::cout << X[i][0] << " " << X[i][1] << " -> " << net.predict(X[i])[0] << "\n";
```

Output (the starting weights are random, so exact numbers vary from run to run):

```
epoch 0    loss 0.327543
epoch 100  loss 0.253182
epoch 300  loss 0.0950404
epoch 500  loss 0.0116156
...
epoch 9900 loss 0.000118317

0 0 -> 0.0104627  (target 0)
0 1 -> 0.989841   (target 1)
1 0 -> 0.98984    (target 1)
1 1 -> 0.0123256  (target 0)
```

## How it works

**Data types.** A vector is `std::vector<double>` (`Vec`) and a matrix is `std::vector<std::vector<double>>` (`Mat`).

**Layers.** Every layer implements two functions:

```cpp
class Layer {
public:
    virtual ~Layer() = default;
    virtual Vec forward(const Vec& input) = 0;
    virtual Vec backward(const Vec& grad_output, double learning_rate) = 0;
};
```

`forward` computes the output and remembers the input. `backward` receives the gradient of the loss with respect to the layer's output, updates the layer's own parameters if it has any, and returns the gradient with respect to its input. That is the chain rule, applied one layer at a time.

**Linear layer.** With weights `W` (out × in) and bias `b`:

```
forward:   y = W·x + b
backward:  dL/dx = Wᵀ · g              (returned to the previous layer)
           dL/dW = g ⊗ x               (outer product)
           dL/db = g
           W -= lr · dL/dW,   b -= lr · g
```

where `g` is the incoming gradient. The returned gradient is computed *before* the weights are updated.

**Activation layer.** Applies a function elementwise; `backward` multiplies the incoming gradient elementwise by the derivative evaluated at the saved input.

**Training (`fit`).** For each epoch and each sample: run `predict`, compute the loss, take the loss derivative as the first gradient, then call `backward` on every layer in reverse order.

## API overview

| Piece | Description |
|---|---|
| `NeuralNetwork::add(unique_ptr<Layer>)` | append a layer |
| `NeuralNetwork::predict(input)` | forward pass through all layers |
| `NeuralNetwork::fit(X, Y, epochs, lr)` | train with per-sample SGD and MSE |
| `Linear(in, out)` | fully connected layer, weights and biases initialised uniformly in [-1, 1) |
| `Activation(f, df)` | elementwise activation from a function and its derivative |
| `sigmoid`, `ReLu` (+ `...Derivative`) | built-in activations |
| `mse`, `mseDerivative` | loss and its gradient |
| `dot`, `outer`, `transpose`, `matVec`, `hadamard`, `add`, `scale`, `substract` | linear algebra helpers |

Custom activation example (leaky ReLU):

```cpp
double alpha = 0.01;
net.add(std::make_unique<Activation>(
    [alpha](double x) { return x > 0 ? x : alpha * x; },
    [alpha](double x) { return x > 0 ? 1.0 : alpha; }));
```

## Known limitations

This version is intentionally simple and favours readability over speed:

- **Performance.** Matrices are `vector<vector<double>>`, almost every operation allocates a new vector, and `Linear::backward` builds a full transposed copy of the weights on every call. Training is single-threaded.
- **Per-sample updates only.** No mini-batches, no shuffling.
- **One loss, two activations.** MSE only; sigmoid and ReLU (any other elementwise function can be passed in). No softmax or cross-entropy.
- **No model saving or loading.**
- **Basic initialisation.** Uniform in [-1, 1), not scaled by layer size.
- **Sigmoid inputs should be small.** Feeding unnormalised, large inputs into sigmoid layers saturates them and training stalls, so normalise your data.
- **Light error checking.** Shape mismatches are not always reported; they can be undefined behaviour.

## Roadmap

A faster rewrite is planned, covering:

- a flat, contiguous matrix type and fewer allocations
- direct transposed multiplication instead of building transposes
- multithreading
- weight export and import
- mini-batches, better initialisation, more losses and activations

## Acknowledgements

Inspired by [SorawitChok/Neural-Network-from-scratch-in-Cpp](https://github.com/SorawitChok/Neural-Network-from-scratch-in-Cpp); the design (`Layer` base class with `forward` and `backward`) follows its README, but all the code here was written independently.
