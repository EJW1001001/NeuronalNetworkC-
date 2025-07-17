#include "neuralnet/Model.h"

/**
 * Default constructor for the Model class.
 */
Model::Model() {}

/**
 * Adds a layer to the model.
 * 
 * @param layer The layer to add.
 */
void Model::addLayer(const Layer& layer) {
    layers.push_back(layer);
}

/**
 * Compiles the model by setting the loss function and optimizer.
 * 
 * @param lossFunction The loss function to use.
 * @param optimizer The optimizer to use for training.
 */
void Model::compile(std::function<double(const std::vector<double>&, const std::vector<double>&)> lossFunction, std::function<void(std::vector<double>&, std::vector<double>&)> optimizer) {
    this->lossFunction = lossFunction;
    this->optimizer = optimizer;
}

/**
 * Trains the model using the provided data.
 * 
 * @param X Input data.
 * @param Y Target data.
 * @param epochs Number of training epochs.
 */
void Model::fit(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            std::vector<double> prediction = predict(X[i]);
            // Calculate error
            std::vector<double> error(Y[i].size());
            for (size_t j = 0; j < Y[i].size(); ++j) {
                error[j] = Y[i][j] - prediction[j];
            }
            // Update weights using the optimizer
            optimizer(prediction, error);
        }
    }
}

/**
 * Makes a prediction using the model.
 * 
 * @param input Input data for prediction.
 * @return The predicted output.
 */
std::vector<double> Model::predict(const std::vector<double>& input) {
    std::vector<double> output = input;
    for (auto& layer : layers) {
        output = layer.forward(output); // forward method in Layer class
    }
    return output;
}
