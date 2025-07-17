#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <cmath>
#include <vector>
#include <functional>

class LossFunctions {
public:
    // Binary Cross Entropy Loss
    static double binaryCrossEntropy(const std::vector<double>& y_true, const std::vector<double>& y_pred);

    // Mean Squared Error Loss
    static double meanSquaredError(const std::vector<double>& y_true, const std::vector<double>& y_pred);
};

#endif // LOSS_FUNCTIONS_H