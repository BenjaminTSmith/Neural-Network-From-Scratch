#include "Layer.h"

int main() {
    std::vector<double> inputs;
    std::vector<double> ground_truth = {0, 1, 0, 0};
    inputs.resize(10);
    std::fill(inputs.begin(), inputs.end(), 0);

    Layer input_layer(10);
    Layer hidden_layer(10, 800);
    Layer output_layer(4, 10);

    input_layer.SetInputs(inputs);

    int i = 1;
    while (output_layer.loss_ > 0.0001 and i < 100) {
        input_layer.ZeroGrad();
        hidden_layer.ZeroGrad();
        output_layer.ZeroGrad();

        std::cout << "epoch " << i << std::endl;
        hidden_layer.ForwardPass(input_layer);
        hidden_layer.ReLU();
        output_layer.ForwardPass(hidden_layer);
        output_layer.SoftMax();
        output_layer.MSE(ground_truth);

        output_layer.BackProp(hidden_layer);
        hidden_layer.BackProp(input_layer);

        hidden_layer.ForwardProp();
        output_layer.ForwardProp();
        i++;
        std::cout << output_layer.loss_ << std::endl;
    }
    std::cout << '\n';
    output_layer.PrintOutput();

    return 0;
}
