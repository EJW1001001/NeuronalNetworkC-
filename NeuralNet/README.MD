# NeuralNet: A Lightweight Neural Network Library in C++

## Overview

NeuralNet is a lightweight neural network library implemented in C++. This project demonstrates the creation of a neural network from scratch, including core components such as neurons, layers, activation functions, loss functions, and optimizers. The library is designed to be simple yet powerful, with optimizations for performance using OpenMP for parallel processing.

## Features

- **Core Components**:
  - Neurons and Layers: Fundamental building blocks of the neural network.
  - Activation Functions: Includes ReLU, Sigmoid, and more.
  - Loss Functions: Binary Cross-Entropy and Mean Squared Error.
  - Optimizers: Adam, SGD, and RMSProp.

- **Parallel Processing**:
  - OpenMP integration for efficient forward propagation in layers.

- **Modular Design**:
  - Easy-to-extend architecture for adding new layers, activation functions, or optimizers.

- **Examples and Tests**:
  - Includes an XOR example to demonstrate the network's functionality.
  - Unit tests to validate the implementation.

## Project Structure

```plaintext
NeuralNet/
├── examples/          # Example usage of the library (e.g., XOR problem)
├── include/
│   └── neuralnet/     # Header files for the library
├── src/               # Source files for the library
├── tests/             # Unit tests for the library
└── CMakeLists.txt     # Build configuration
```

## Getting Started

### Prerequisites

- C++17 or later
- CMake 3.10 or later
- OpenMP support

### Build Instructions

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd NeuralNet
   ```

2. Create a build directory and configure the project:

   ```bash
   mkdir build
   cd build
   cmake ..
   ```

3. Build the project:

   ```bash
   cmake --build .
   ```

### Running Examples

To run the XOR example:

```bash
./examples/xor_example
```

To run the unit tests:

```bash
./tests/test_neuralnet
```

## Implementation Details

### Neurons and Layers

- Each neuron performs a weighted sum of its inputs and applies an activation function.
- Layers manage multiple neurons and handle forward propagation.

### Activation Functions

- Implemented as reusable components, allowing dynamic selection during runtime.

### Loss Functions

- Binary Cross-Entropy: Suitable for classification tasks.
- Mean Squared Error: Suitable for regression tasks.

### Optimizers

- Adam: Combines the advantages of RMSProp and Momentum.
- SGD: Stochastic Gradient Descent.
- RMSProp: Adaptive learning rate optimization.

### OpenMP Integration

- Forward propagation in layers is parallelized using OpenMP, significantly improving performance for large networks.

## Future Work

- Add support for GPU acceleration.
- Implement additional activation functions and optimizers.
- Expand the library with more complex examples.

