#ifndef LAYER_H
#define LAYER_H

#include <cmath>

#include "neuron.h"

struct Layer {
    std::vector<Neuron> neurons_;
    const double alpha_ = 0.1;

    Layer(int nneurons, int nins) : neurons_(nneurons) {
        for (auto& neuron_ : neurons_) neuron_ = Neuron(nins);
    }

    void ForwardProp(Layer& in) {
        for (auto& neuron_ : neurons_)
            neuron_.ForwardPass(in.neurons_);
    }

    void ForwardProp(const Matrix<Dval>& in) {
        for (auto& neuron_ : neurons_) neuron_.ForwardPass(in);
    }

    void BackProp(Layer& in) {
        for (auto& neuron_ : neurons_)
            neuron_.BackProp(in.neurons_, alpha_);
    }

    void BackProp(const Matrix<Dval>& in) {
        for (auto& neuron_ : neurons_)
            neuron_.BackProp(in, alpha_);
    }

    void Activate() {
        for (auto& neuron_ : neurons_)
            neuron_.Activate();
    }

    void ZeroGrad() {
        for (auto& neuron_ : neurons_)
            neuron_.ZeroGrad();
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
    layer_mat = layer_mat.SoftMax();

    loss = (layer_mat - ground_truth).Square().Average().value_;

    for (int i = 0; i < layer.neurons_.size(); i++) {
        for (int j = 0; j < ground_truth.col_count_; j++) {
            layer.neurons_[i].out_[j].grad_ += 2 *
                (layer_mat[i * layer_mat.col_count_ + j] -
                ground_truth[i * ground_truth.col_count_ + j].value_) /
                ground_truth.size() * layer_mat[i * layer_mat.col_count_ + j] *
                (1 - layer_mat[i * layer_mat.col_count_ + j]) *
                (layer.neurons_[i].out_[j].value_ > 0);
        }
    }

    return loss;
}

static double GetCorrectGuesses(Layer& layer, const Matrix<Dval>& ground_truth) {
    std::vector<Dval> elements;
    for (auto& neuron_ : layer.neurons_) {
        for (auto& val : neuron_.out_.elements_) {
            elements.push_back(val);
        }
    }
    Matrix<Dval> layer_mat(layer.neurons_.size(),
                           layer.neurons_[0].out_.size());
    layer_mat.SetElements(elements);
    
    std::vector<double> correct_guesses;
    std::vector<double> nn_guesses;
    for (int i = 0; i < ground_truth.col_count_; i++) {
        for (int j = 0; j < ground_truth.row_count_; j++) {
            if (ground_truth[j * ground_truth.col_count_ + i] == 1) {
                correct_guesses.push_back(j);
                break;
            }
        }
    }
    for (int i = 0; i < layer_mat.col_count_; i++) {
        double max = 0;
        int max_idx = 0;
        for (int j = 0; j < layer_mat.row_count_; j++) {
            if (layer_mat[j * layer_mat.col_count_ + i] > max) {
                max = layer_mat[j * layer_mat.col_count_ + i].value_;
                max_idx = j;
            }
        }
        nn_guesses.push_back(max_idx);
    }

    double result = 0;
    for (int i = 0; i < correct_guesses.size(); i++) {
        if (correct_guesses[i] == nn_guesses[i]) result++;
    }
    return result;
}

#endif // !LAYER_H
