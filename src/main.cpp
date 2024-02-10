#include "NN.h"
#include "LoadCSV.h"

int main() {

    // place mnist dataset or any csv within build directory
    /*auto inputs = parseCSV("mnist_train.csv");
    std::cout << inputs[0].size() << std::endl;
    std::cout << inputs[0][783] << std::endl;
    std::cout << inputs[0][784] << std::endl;*/
    
    auto neural_network = new NeuralNetwork<double>(5, {10}, 10);
    neural_network->set_input_layer({ 0.5, 3, 10, -3, 4 });
    std::vector<double> ground_truth = {
        1, -1, 0.3, -0.75, -1, 0.5, -0.3, -0.5, 0.5, 0.83,
    };

    neural_network->ForwardPass();

    std::cout << "Predictions: " << std::endl;
    neural_network->PrintOutput();

    for (int i = 0; i < 100000; ++i) {
        neural_network->Pass(ground_truth);
    }

    std::cout << "Final Predictions: " << std::endl;
    neural_network->PrintOutput();

    delete neural_network;

    return 0;
}
