#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <vector>

class Optimizers {
public:
    static void adam(std::vector<double>& weights, std::vector<double>& gradients);
    static void sgd(std::vector<double>& weights, std::vector<double>& gradients, double learningRate);
    static void rmsprop(std::vector<double>& weights, std::vector<double>& gradients, double learningRate);
};

#endif // OPTIMIZERS_H
