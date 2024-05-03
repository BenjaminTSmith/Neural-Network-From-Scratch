#include <iostream>

#include "file_reader.h"
#include "layer.h"

int main() {
    int batch_size = 2;
    int rows = 500;
    int epochs = 10000;
    int max_accuracy = 95;

    Layer input_layer(50, 784);
    Layer output_layer(10, 50);

    std::cout << "Training settings:\n" << "Batch size: " << batch_size <<
        "\nTraining data size: " << rows << "\nLearning rate: " << 
        output_layer.alpha_ << "\nEpochs: " << epochs << "\n";

    auto inputs = ReadData("mnist_train.csv", rows, batch_size);
    auto test_data = ReadData("mnist_test.csv", 10000, 1);


    for (int i = 0; i < epochs; i++) {
        double total_loss = 0;
        double correct_guesses = 0;
        for (const auto& input : inputs) {
            input_layer.ForwardProp(input.image_data);
            input_layer.Activate();
            output_layer.ForwardProp(input_layer);
            output_layer.Activate();
            total_loss += ComputeLoss(output_layer, input.label);
            output_layer.BackProp(input_layer);
            input_layer.BackProp(input.label);
            input_layer.ZeroGrad();
            output_layer.ZeroGrad();
            correct_guesses += GetCorrectGuesses(output_layer,
                                                       input.label);
        }
        std::cout << "epoch: " << i << "; Average loss: " 
            << total_loss / ((float)rows / batch_size) << "; Accuracy: " 
            << correct_guesses / rows * 100 << "%\n";
        if (correct_guesses / rows * 100 > max_accuracy)
            break;
    }

    double correct_guesses = 0;
    for (const auto& input : test_data) {
        input_layer.ForwardProp(input.image_data);
        input_layer.Activate();
        output_layer.ForwardProp(input_layer);
        output_layer.Activate();
        correct_guesses += GetCorrectGuesses(output_layer,
                                             input.label);
    }
    std::cout << "Test Data: " <<  "Accuracy: " <<
        correct_guesses / 10000 * 100 << "%\n";
    return 0;
}
