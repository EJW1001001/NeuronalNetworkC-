#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <functional>
#include "neuralnet/Layer.h"
#include "neuralnet/Activations.h"
#include "neuralnet/LossFunctions.h"

class Model {
private:
    std::vector<Layer> layers;
    std::function<double(const std::vector<double>&, const std::vector<double>&)> lossFunction;
    std::function<void(std::vector<double>&, std::vector<double>&)> optimizer;

public:
    Model();
    void addLayer(const Layer& layer);
    void compile(std::function<double(const std::vector<double>&, const std::vector<double>&)> lossFunction, std::function<void(std::vector<double>&, std::vector<double>&)> optimizer);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, int epochs);
    std::vector<double> predict(const std::vector<double>& input);
};

#endif // MODEL_H
