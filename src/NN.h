#ifndef NN_H
#define NN_H

#include "Layer.h"

template<typename T>
class NeuralNetwork {
private:
    const int layerCount;
public:
    Layer<T> inputLayer;
    std::vector<Layer<Neuron>> layers;
    Layer<Neuron> outputLayer;

    NeuralNetwork(int inputLayer, const std::vector<int>& hiddenLayers,
                  int outputLayer)
        : inputLayer(inputLayer, 0),
          layerCount(hiddenLayers.size() + 2),
          outputLayer(outputLayer, hiddenLayers[hiddenLayers.size() - 1])
    {
        layers.emplace_back(hiddenLayers[0], inputLayer);
        for (int i = 1; i < hiddenLayers.size(); i++) {
            layers.emplace_back(hiddenLayers[i], hiddenLayers[i - 1]);
        }
    }

    void forwardPass() {
        layers[0].forwardPass(inputLayer);
        for (int i = 1; i < layerCount - 2; i++) {
            layers[i].forwardPass(layers[i - 1]);
        }
        outputLayer.forwardPass(layers[layers.size() - 1]);
    }


    int getLayerCount() { return layerCount; }

};



#endif // NN_H
