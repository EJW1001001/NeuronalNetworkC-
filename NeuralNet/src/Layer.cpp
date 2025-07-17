#include "neuralnet/Layer.h"

/**
 * Constructor for the Layer class.
 * Initializes the layer with the specified number of neurons and inputs.
 * 
 * @param num_neurons Number of neurons in the layer.
 * @param num_inputs Number of inputs to each neuron.
 * @param activationFunc Activation function for the neurons.
 */
Layer::Layer(int num_neurons, int num_inputs, std::function<double(double)> activationFunc) : activationFunction(activationFunc) {
    for (int i = 0; i < num_neurons; ++i) {
        neurons.emplace_back(num_inputs);
    }
}

/**
 * Performs forward propagation through the layer.
 * 
 * @param inputs Input data for the layer.
 * @return The output of the layer after forward propagation.
 */
std::vector<double> Layer::forward(const std::vector<double>& inputs) {
    std::vector<double> outputs(neurons.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < neurons.size(); ++i) {
        outputs[i] = neurons[i].activate(inputs, activationFunction);
    }

    return outputs;
}
