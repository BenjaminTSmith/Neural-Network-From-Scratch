#include <iostream>
#include "Layer.h"

int main() {
    std::vector<double> inputs;
    std::vector<double> ground_truth = {0, 0, 0, 0, 1, 0, 0};
    inputs.resize(800);
    std::fill(inputs.begin(), inputs.end(), 0);

    Layer input_layer(800);
    Layer hidden_layer(10, 800);
    Layer output_layer(7, 10);

    input_layer.SetInputLayer(inputs);

    while (output_layer.loss_ > 0.00001) {
        input_layer.ZeroGrad();
        hidden_layer.ZeroGrad();
        output_layer.ZeroGrad();

        // std::cout << "epoch " << i << std::endl;
        hidden_layer.ForwardPass(input_layer);
        hidden_layer.ReLU();
        output_layer.ForwardPass(hidden_layer);
        output_layer.SoftMax();
        output_layer.SparseCategoricalCrossEntropy(ground_truth);

        output_layer.BackProp(hidden_layer);
        hidden_layer.BackProp(input_layer);

        hidden_layer.ForwardProp();
        output_layer.ForwardProp();
        std::cout << output_layer.loss_ << std::endl;
    }
    std::cout << '\n';
    output_layer.PrintOutput();

    return 0;
}
