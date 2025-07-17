#include "neuralnet/Activations.h"

/**
 * Sigmoid activation function.
 * 
 * @param x Input value.
 * @return The result of applying the sigmoid function.
 */
double ActivationFunctions::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * ReLU activation function.
 * 
 * @param x Input value.
 * @return The result of applying the ReLU function.
 */
double ActivationFunctions::relu(double x) {
    return x > 0 ? x : 0;
}

/**
 * Tanh activation function.
 * 
 * @param x Input value.
 * @return The result of applying the tanh function.
 */
double ActivationFunctions::tanh(double x) {
    return std::tanh(x);
}

/**
 * Softmax activation function for a specific index.
 * 
 * @param inputs Vector of input values.
 * @param index Index for which to calculate the softmax value.
 * @return The softmax value for the specified index.
 */
double ActivationFunctions::softmax(const std::vector<double>& inputs, size_t index) {
    double sum_exp = 0.0;
    for (double val : inputs) {
        sum_exp += std::exp(val);
    }
    return std::exp(inputs[index]) / sum_exp;
}