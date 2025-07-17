#include "neuralnet/Model.h"
#include "neuralnet/Layer.h"
#include "neuralnet/Activations.h"
#include "neuralnet/LossFunctions.h"
#include "neuralnet/Optimizers.h"
#include <iostream>
#include <vector>

// Example of using the neural network to solve XOR
int main() {
    // Create a model
    Model model;

    // Add layers with specific activation functions
    model.addLayer(Layer(2, 2, ActivationFunctions::relu));
    model.addLayer(Layer(1, 2, ActivationFunctions::sigmoid));

    // Compile the model
    model.compile(LossFunctions::binaryCrossEntropy, Optimizers::adam);

    // Training data for XOR
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> Y = {{0}, {1}, {1}, {0}};

    // Train the model
    model.fit(X, Y, 1000);

    // Test the model
    for (const auto& input : X) {
        std::vector<double> output = model.predict(input);
        std::cout << "Input: " << input[0] << ", " << input[1] << " -> Output: " << output[0] << std::endl;
    }

    return 0;
}
