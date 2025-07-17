#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <functional>
#include "neuralnet/Activations.h"

class Neuron {
public:
    std::vector<double> weights;
    double bias;

    Neuron(int num_inputs);
    double activate(const std::vector<double>& inputs, std::function<double(double)> activationFunc);

};

#endif // NEURON_H