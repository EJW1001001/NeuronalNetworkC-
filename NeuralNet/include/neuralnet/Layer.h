#ifndef LAYER_H
#define LAYER_H

#include "neuralnet/Neuron.h"
#include <vector>
#include <functional>

class Layer {
private:
    std::vector<Neuron> neurons;
    std::function<double(double)> activationFunction;

public:
    Layer(int num_neurons, int num_inputs, std::function<double(double)> activationFunc);
    std::vector<double> forward(const std::vector<double>& inputs);
};

#endif // LAYER_H