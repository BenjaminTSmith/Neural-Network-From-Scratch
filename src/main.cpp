#include "NN.h"
#include "LoadCSV.h"

int main() {

    // auto inputs = parseCSV("mnist_train.csv");
    
    auto neural_network = new NeuralNetwork<double>(5, {5}, 12);
    neural_network->set_input_layer({ 0.5, 3, 10, -3, 4 });
    std::vector<double> ground_truth = {
        1, -1, 0.3, -0.75, -1, 0.5, -0.3, -0.5, 0.5, 0.83, 0.0003, 0.000098
    };

    neural_network->ForwardPass();

    std::cout << "Predictions: " << std::endl;
    neural_network->PrintOutput();

    for (int i = 0; i < 100000; ++i) {
        neural_network->BackProp(ground_truth);
        neural_network->ForwardProp();
        neural_network->ForwardPass();
    }

    std::cout << "Final Predictions: " << std::endl;
    neural_network->PrintOutput();

    delete neural_network;

    return 0;
}
