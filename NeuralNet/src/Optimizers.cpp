#include "neuralnet/Optimizers.h"
#include <cmath>

/**
 * Implements the Adam optimization algorithm.
 * 
 * @param weights Vector of weights to be updated.
 * @param gradients Vector of gradients for the weights.
 */
void Optimizers::adam(std::vector<double>& weights, std::vector<double>& gradients) {
    static std::vector<double> m(weights.size(), 0.0);
    static std::vector<double> v(weights.size(), 0.0);
    static int t = 0;

    double beta1 = 0.9;
    double beta2 = 0.999;
    double learningRate = 0.001;
    double epsilon = 1e-8;

    t++;

    for (size_t i = 0; i < weights.size(); ++i) {
        m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
        v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i];

        double m_hat = m[i] / (1 - std::pow(beta1, t));
        double v_hat = v[i] / (1 - std::pow(beta2, t));

        weights[i] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

/**
 * Implements the Stochastic Gradient Descent (SGD) optimization algorithm.
 * 
 * @param weights Vector of weights to be updated.
 * @param gradients Vector of gradients for the weights.
 * @param learningRate Learning rate for the optimization.
 */
void Optimizers::sgd(std::vector<double>& weights, std::vector<double>& gradients, double learningRate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learningRate * gradients[i];
    }
}

/**
 * Implements the RMSProp optimization algorithm.
 * 
 * @param weights Vector of weights to be updated.
 * @param gradients Vector of gradients for the weights.
 * @param learningRate Learning rate for the optimization.
 */
void Optimizers::rmsprop(std::vector<double>& weights, std::vector<double>& gradients, double learningRate) {
    static std::vector<double> cache(weights.size(), 0.0);
    double decayRate = 0.9;
    double epsilon = 1e-8;

    for (size_t i = 0; i < weights.size(); ++i) {
        cache[i] = decayRate * cache[i] + (1 - decayRate) * gradients[i] * gradients[i];
        weights[i] -= learningRate * gradients[i] / (std::sqrt(cache[i]) + epsilon);
    }
}
