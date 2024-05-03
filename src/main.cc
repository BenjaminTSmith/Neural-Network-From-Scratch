#include <iostream>
#include "layer.h"

int main() {

    Layer input_layer(10, 5);
    Layer hidden_layer(10, 10);
    Layer output_layer(5, 10);

    Matrix<Dval> input(5, 1);
    input.SetElements({ 0, 0.5, 0.02, 0, 0.3});
    Matrix<Dval> ground_truth(5, 1);
    ground_truth.SetElements({ 0, 0, 0.5, 0.5, 0});

    for (int i = 0; i < 150; i++) {
        input_layer.ForwardProp(input);
        hidden_layer.ForwardProp(input_layer);
        output_layer.ForwardProp(hidden_layer);
        double loss = ComputeLoss(output_layer, ground_truth);
        std::cout << "epoch: " << i << "; Loss: " << loss << "\n";
        if (loss < 0.001)
            break;
        output_layer.BackProp(hidden_layer);
        hidden_layer.BackProp(input_layer);
        input_layer.BackProp(input);
        input_layer.ZeroGrad();
        hidden_layer.ZeroGrad();
        output_layer.ZeroGrad();
    }

    std::cout << output_layer << "\n";

    return 0;
}


