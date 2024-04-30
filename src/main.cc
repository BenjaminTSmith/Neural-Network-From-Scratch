#include <iostream>
#include "layer.h"

int main() {
    
    Layer input_layer(2, 2);
    Layer output_layer(3, 2);

    Matrix<Dval> mat(2, 3);
    mat.SetElements({ 0.5, 0.5, 0.1, 0.9, 0.2, 1.0 });
    Matrix<Dval> ground_truth(3, 3);
    ground_truth.SetElements({ 0, 0.8, 0.3, 0, 0.9, 0.2, 1, 0, 0 });


    for (int i = 0; i < 100000; i++) {
        input_layer.ForwardProp(mat);
        output_layer.ForwardProp(input_layer);
        double loss;
        std::cout << (loss = ComputeLoss(output_layer, ground_truth)) << "\n";
        if (loss < 0.001)
            break;
        output_layer.BackProp(input_layer);
        input_layer.BackProp(mat);
        input_layer.ZeroGrad();
        output_layer.ZeroGrad();
    }
    std::cout << "\n" << output_layer;

    return 0;
}


