#ifndef NN_H
#define NN_H

#include "Layer.h"

template<typename T, typename U>
class NeuralNetwork {
private:
    int layerCount;
public:
    Layer<T> inputLayer;
    std::vector<Layer<U>> layers;

    NeuralNetwork(int layerCount, const std::vector<int>& layerSizes)
        : layerCount(layerCount)
    {
        layers.emplace_back(layerSizes[0], 0);
        for (int i = 1; i < layerSizes.size(); i++) {
        }
    }


};



#endif // NN_H
