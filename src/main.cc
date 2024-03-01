#include <iostream>
#include "Layer.h"

int main() {

    Eigen::MatrixXd input(1, 3);
    input << 0, 1, 2;
    std::cout << input << std::endl;
    Eigen::MatrixXd ground_truth(1, 3);
    ground_truth << 3, -2, -1;

    nn::Layer hidden_layer(5, 3);
    nn::Layer output_layer(1, 5);

    hidden_layer.ForwardPass(input);
    hidden_layer.PrintOutput();
    output_layer.ForwardPass(output_layer);
    output_layer.PrintOutput();

    return 0;
}
