#ifndef NN_H
#define NN_H

#include "Layer.h"
#include <cmath>

template<typename T>
class NeuralNetwork {
private:
    const int layer_count_;
    Layer<T> input_layer_;
public:
    std::vector<Layer<Neuron>> layers_;
    Layer<Neuron> output_layer_;

    void set_input_layer(const std::vector<T>& inputs) {
        input_layer_.neurons_ = inputs;
    }

    NeuralNetwork(int input_layer, const std::vector<int>& hidden_layers,
                  int output_layer)
        : input_layer_(input_layer),
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

    void BackProp(const std::vector<double>& ground_truth) {
        ZeroGrad();
        MSELoss(ground_truth);
        output_layer_.BackProp(layers_[layers_.size() - 1]);
        for (size_t i = layers_.size() - 1; i > 0; i--) {
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

    void Prop(std::vector<double> ground_truth) {
        BackProp(ground_truth);
        ForwardProp();
        // ForwardPass();
    }

    double MSELoss(const std::vector<double>& ground_truth) {
        double MSE = 0;
        for (int i = 0; i < ground_truth.size(); i++) {
            MSE += std::pow(output_layer_.neurons_[i].out_.value_ 
                            - ground_truth[i], 2);
            output_layer_.neurons_[i].out_.grad_ = 2 * 
                (output_layer_.neurons_[i].out_.value_ - ground_truth[i]);
        }
        return (1 / static_cast<double>(ground_truth.size())) * MSE;
    }

    int layer_count() { return layer_count_; }

    void PrintOutput() {
        for (const auto& neuron: output_layer_.neurons_) {
            std::cout << neuron.out_.value_ << std::endl;
        }
        std::cout << std::endl;
    }

    int SoftMax() {
        // Not actual soft max. Just a test right now
        int index = -1;
        int max = -1;
        for (int i = 0; i < output_layer_.neurons_.size(); i++) {
            if (output_layer_.neurons_[i].out_.value_ > max) {
                max = output_layer_.neurons_[i].out_.value_;
                index = i;
            }
        }
        return index;
    }
};

#endif // NN_H
