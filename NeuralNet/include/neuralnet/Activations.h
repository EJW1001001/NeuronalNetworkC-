#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>
#include <cmath>

class ActivationFunctions {
public:
    static double sigmoid(double x);
    static double relu(double x);
    static double tanh(double x);
    static double softmax(const std::vector<double>& inputs, size_t index);
};

#endif // ACTIVATIONS_H