#include "neuralnet/LossFunctions.h"
#include <cmath>

/**
 * Calculates the Binary Cross-Entropy loss.
 * 
 * @param y_true Vector of true labels.
 * @param y_pred Vector of predicted labels.
 * @return The calculated loss.
 */
double LossFunctions::binaryCrossEntropy(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        loss += -y_true[i] * std::log(y_pred[i]) - (1 - y_true[i]) * std::log(1 - y_pred[i]);
    }
    return loss / y_true.size();
}

/**
 * Calculates the Mean Squared Error (MSE) loss.
 * 
 * @param y_true Vector of true labels.
 * @param y_pred Vector of predicted labels.
 * @return The calculated loss.
 */
double LossFunctions::meanSquaredError(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        loss += std::pow(y_true[i] - y_pred[i], 2);
    }
    return loss / y_true.size();
}
