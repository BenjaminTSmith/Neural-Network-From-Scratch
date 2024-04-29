#include <iostream>
#include "layer.h"

int main() {
    
    Layer input_layer(5, 5);
    Layer hidden_layer(10, 5);
    Layer output_layer(10, 10);

    Matrix<Dval> mat(5, 1);
    mat.SetElements({ 3, -1, 1, 1, 2 });
    Matrix<Dval> ground_truth(10, 1);
    ground_truth.SetElements({ 4, 1, 8, 9, 10, 237, 0.05, 42.8, 0.3, 3999 });


    for (int i = 0; i < 10000; i++) {
        input_layer.ForwardProp(mat);
        hidden_layer.ForwardProp(input_layer);
        output_layer.ForwardProp(hidden_layer);
        double loss;
        std::cout << (loss = ComputeLoss(output_layer, ground_truth)) << "\n";
        if (loss < 0.001)
            break;
        output_layer.BackProp(hidden_layer);
        hidden_layer.BackProp(input_layer);
        input_layer.BackProp(mat);
        input_layer.ZeroGrad();
        hidden_layer.ZeroGrad();
        output_layer.ZeroGrad();
    }

    std::cout << output_layer;

    return 0;
}


