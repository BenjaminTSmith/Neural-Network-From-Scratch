#include "Layer.h"
#include "CSVParser.h"

int main() {

    int epochs = 10;
    int batch_size = 64;
    std::vector<Image> images = ParseCSV("mnist_test.csv");

    Layer input_layer(784);
    Layer hidden_layer1(10, 784);
    Layer hidden_layer2(10, 10);
    Layer output_layer(10, 10);

    for (int i = 0; i < epochs; ++i) {
        std::cout << "epoch: " << i << std::endl;
        for (int j = 0; j < images.size(); j++) {
            input_layer.SetInputLayer(images[j].data);

            hidden_layer1.ForwardPass(input_layer);
            hidden_layer1.LeakyReLU();
            hidden_layer2.ForwardPass(hidden_layer1);
            hidden_layer2.LeakyReLU();
            output_layer.ForwardPass(hidden_layer2);
            output_layer.SoftMax();
            output_layer.MSE(images[j].one_hot_label);

            output_layer.BackProp(hidden_layer2);
            hidden_layer2.BackProp(hidden_layer1);
            hidden_layer1.BackProp(input_layer);

            auto output = output_layer.GetOutput();
            int max = 0;
            int choice = 0;
            for (int k = 0; k < output.size(); ++k) {
                if (output[k] > max) {
                    choice = k;
                    max = output[k];
                }
            }

            // output_layer.PrintOutput();

            // std::cout << output_layer.loss_ << std::endl;
            if (j % batch_size == 0) {
                output_layer.AverageGrad(batch_size);
                hidden_layer1.AverageGrad(batch_size);
                hidden_layer2.AverageGrad(batch_size);
                hidden_layer2.ForwardProp();
                hidden_layer1.ForwardProp();
                output_layer.ForwardProp();

                input_layer.ZeroGrad();
                hidden_layer1.ZeroGrad();
                hidden_layer2.ZeroGrad();
                output_layer.ZeroGrad();
                std::cout << "batch loss: " << output_layer.loss_ << std::endl;
            }
        }
    }

    std::cout << '\n';

    output_layer.PrintOutput();

    return 0;
}
