#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

struct Layer {
    std::vector<Neuron> neurons_;

    Layer(int nneurons, int nins) : neurons_(nneurons) {
        for (auto& neuron_ : neurons_) neuron_ = Neuron(nins);
    }

    void ForwardProp(Layer& in) {
        for (auto& neuron_ : neurons_)
            neuron_.ForwardProp(in.neurons_);
    }

    void ForwardProp(const Matrix<Dval>& in) {
        for (auto& neuron_ : neurons_) neuron_.ForwardProp(in);
    }

    void BackProp(Layer& in) {
        for (auto& neuron_ : neurons_)
            neuron_.BackProp(in.neurons_);
    }

    void BackProp(const Matrix<Dval>& in) {
        for (auto& neuron_ : neurons_)
        neuron_.BackProp(in);
    }

    void SetGrad(int val=1) {
        for (auto& neuron : neurons_) {
            for (auto& out : neuron.out_.elements_) {
                out.grad_ = 1;
            }
        }
    }
};

static std::ostream& operator<<(std::ostream& oss, const Layer& layer) {
    for (const auto& neuron_ : layer.neurons_)
        std::cout << neuron_.out_ << '\n';
    return oss;
}

static double ComputeLoss(Layer& layer,
                          const std::vector<double>& ground_truth) {
    
    return 0;
}

static void ComputeLossDerivative(Layer& layer,
                                  const std::vector<double>& ground_truth) {

}

#endif // !LAYER_H
