#include "NN.h"
#include "LoadCSV.h"
#include <ostream>

int main() {

    // place mnist dataset or any csv within build directory
    auto train_data = ParseCSV("mnist_train.csv");
    std::cout << "Done parsing training data!" << std::endl;

    auto test_data = ParseCSV("mnist_test.csv");
    std::cout << "Done parsing testing data!" << std::endl;
    
    auto neural_network = new NeuralNetwork<double>(784, {10}, 10);
    

    for (int i = 0; i < 2; ++i) {
        std::cout << "epoch " << i << std::endl;
        double right = 0;
        double wrong = 0;
        for (const auto& input: train_data) {
            std::vector<double> ground_truth = GetGroundTruth(input);
            int answer = input[0];
            neural_network->set_input_layer(input);
            neural_network->ForwardPass();
             
            if (neural_network->SoftMax() == answer) { right++; }
            else { wrong++; }
            neural_network->Prop(ground_truth);
        }
        std::cout << "epoch " << i << " done." << std::endl;
        std::cout << "Accuracy: " << right / (right + wrong) * 100 << '%' << std::endl;
    }

    std::cout << "Starting test data" << std::endl;
    double right = 0;
    double wrong = 0;
    for (auto& input: test_data) {
        std::vector<double> ground_truth = GetGroundTruth(input);
        int answer = input[0];
        neural_network->set_input_layer(input);
        neural_network->ForwardPass();

        if (neural_network->SoftMax() == answer) { right++; }
        else { wrong++; }
        neural_network->Prop(ground_truth);
    }
    std::cout << "Test accuracy: " << right / (right + wrong) * 100 << '%' << std::endl;

    delete neural_network;

    return 0;
}
