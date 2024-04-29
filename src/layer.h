#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

struct Layer {
    std::vector<Neuron> neurons_;
    const double alpha_ = 0.1;

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
            neuron_.BackProp(in.neurons_, alpha_);
    }

    void BackProp(const Matrix<Dval>& in) {
        for (auto& neuron_ : neurons_)
            neuron_.BackProp(in, alpha_);
    }

    void SetGrad(int val=1) {
        for (auto& neuron : neurons_) {
            for (auto& out : neuron.out_.elements_) {
                out.grad_ = 1;
            }
        }
    }

    void ZeroGrad() {
        for (auto& neuron_ : neurons_)
            neuron_.ZeroGrad();
    }

    Layer SoftMax() const {
        //TODO
        return *this;
    }
};

static std::ostream& operator<<(std::ostream& oss, const Layer& layer) {
    for (const auto& neuron_ : layer.neurons_)
        std::cout << neuron_.out_ << '\n';
    return oss;
}

static double ComputeLoss(Layer& layer,
                          const Matrix<Dval>& ground_truth) {
    double loss;
    std::vector<Dval> elements;
    for (auto& neuron_ : layer.neurons_) {
        for (auto& val : neuron_.out_.elements_) {
            elements.push_back(val);
        }
    }
    Matrix<Dval> layer_mat(layer.neurons_.size(),
                           layer.neurons_[0].out_.size());
    layer_mat.SetElements(elements);

    loss = (layer_mat - ground_truth).Square().Average().value_;

    for (int i = 0; i < layer.neurons_.size(); i++) {
        for (int j = 0; j < ground_truth.col_count_; j++) {
            layer.neurons_[i].out_[j].grad_ += 2 *
                (layer.neurons_[i].out_[j].value_ -
                ground_truth[i * ground_truth.col_count_ + j].value_) /
                ground_truth.size() * layer.alpha_;
        }
    }

    return loss;
}

#endif // !LAYER_H
