#ifndef NN_H
#define NN_H

#include "Layer.h"

template<typename T>
class NeuralNetwork {
private:
    const int layer_count_;
public:
    Layer<T> input_layer_;
    std::vector<Layer<Neuron>> layers_;
    Layer<Neuron> output_layer_;

    NeuralNetwork(int input_layer, const std::vector<int>& hidden_layers,
                  int output_layer)
        : input_layer_(input_layer, 0),
          layer_count_(hidden_layers.size() + 2),
          output_layer_(output_layer, hidden_layers[hidden_layers.size() - 1])
    {
        layers_.emplace_back(hidden_layers[0], input_layer);
        for (int i = 1; i < hidden_layers.size(); i++) {
            layers_.emplace_back(hidden_layers[i], hidden_layers[i - 1]);
        }
    }

    void ForwardPass() {
        layers_[0].ForwardPass(input_layer_);
        for (int i = 1; i < layer_count_ - 2; i++) {
            layers_[i].ForwardPass(layers_[i - 1]);
        }
        output_layer_.ForwardPass(layers_[layers_.size() - 1]);
    }

    void BackProp() {
        ZeroGrad();
        for (auto& neuron: output_layer_.neurons_) {
            neuron.out_.grad_ = 1;
        }
        output_layer_.BackProp(layers_[layers_.size() - 1]);
        for (size_t i = layers_.size() - 1; i > 0; i++) {
            layers_[i].BackProp(layers_[i - 1]);
        }
        layers_[0].BackProp(input_layer_);
    }

    void ZeroGrad() {
        for (auto& layer: layers_) {
            layer.ZeroGrad();
        }
        output_layer_.ZeroGrad();
    }

    void ForwardProp() {
        for (auto& layer: layers_) {
            layer.ForwardProp();
        }
        output_layer_.ForwardProp();
    }

    int GetLayerCount() { return layer_count_; }

};



#endif // NN_H
