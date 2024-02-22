#include "Layer.h"

int main() {
    std::vector<double> ground_truth = {3, 1, 5};
    std::vector<double> inputs = {-4, 23, 7};

    Layer input_layer(3);
    Layer hidden_layer(5, 3);
    Layer output_layer(3, 5);

    input_layer.SetInputs(inputs);

    for (int i = 0; i < 3; i++) {
        input_layer.ZeroGrad();
        hidden_layer.ZeroGrad();
        output_layer.ZeroGrad();

        std::cout << "epoch " << i << std::endl;
        hidden_layer.ForwardPass(input_layer);
        hidden_layer.ReLU();
        output_layer.ForwardPass(hidden_layer);
        output_layer.MSE(ground_truth);

        output_layer.BackProp(hidden_layer);
        hidden_layer.BackProp(input_layer);

        hidden_layer.ForwardProp();
        output_layer.ForwardProp();
        std::cout << output_layer.loss_ << std::endl;
        output_layer.PrintOutput();
    }


    return 0;
}
