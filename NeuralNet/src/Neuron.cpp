#include "neuralnet/Neuron.h"

/**
 * Constructor for the Neuron class.
 * Initializes the weights and bias for the neuron.
 * 
 * @param num_inputs Number of inputs to the neuron.
 */
Neuron::Neuron(int num_inputs)
{
    weights.resize(num_inputs);

    for (auto &w : weights)
        w = ((rand() % 100) / 100.0 - 0.5);
    bias = 0.0;
}

/**
 * Activates the neuron by applying the activation function to the weighted sum of inputs.
 * 
 * @param inputs Vector of input values.
 * @param activationFunc Activation function to apply.
 * @return The output of the neuron after applying the activation function.
 */
double Neuron::activate(const std::vector<double> &inputs, std::function<double(double)> activationFunc)
{
    double z = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i)
        z += weights[i] * inputs[i];
    z += bias;
     return activationFunc(z);
}