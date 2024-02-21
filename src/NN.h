#ifndef NN_H
#define NN_H

#include "Layer.h"
#include <cmath>
#include <iostream>

class NeuralNetwork {
private:
    const int layer_count_;
    std::vector<Layer> layers_;
    double loss_ = 0;
public:

    std::vector<Layer> layers() { return layers_; }
    int layer_count() { return layer_count_; }
    double loss() { return loss_; }

    void set_input_layer(const std::vector<Neuron>& inputs) { 
        layers_[0].neurons_ = inputs; 
    }

    NeuralNetwork(int input_layer, const std::vector<int>& hidden_layers,
                  int output_layer)
          : layer_count_(hidden_layers.size() + 2) {
        layers_.emplace_back(hidden_layers[0], input_layer);
        for (int i = 1; i < hidden_layers.size(); i++) {
            layers_.emplace_back(hidden_layers[i], hidden_layers[i - 1]);
        }
    }

    void ForwardPass() {
        for (int i = 1; i < layer_count_ - 2; i++) {
            layers_[i].ForwardPass(layers_[i - 1]);
        }
    }

    void BackProp(const std::vector<double>& ground_truth) {
        for (size_t i = layers_.size() - 1; i > 0; i--) {
            layers_[i].BackProp(layers_[i - 1]);
        }
    }

    void ZeroGrad() {
        for (auto& layer: layers_) {
            layer.ZeroGrad();
        }
    }

    void ForwardProp() {
        for (auto& layer: layers_) {
            layer.ForwardProp();
        }
    }

    Layer& operator[](int index) {
        return layers_[index];
    }
};

#endif // NN_H
