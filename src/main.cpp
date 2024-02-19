#include "NN.h"
#include "LoadCSV.h"
#include "Neuron.h"

int ReturnSelection(const Layer<Neuron>& output_layer) {
    double max = 0;
    int index = 0;
    for (int i = 0; i < output_layer.neurons_.size(); ++i) {
        if (output_layer.neurons_[i].out_.value_ > max) {
            max = output_layer.neurons_[i].out_.value_;
            index = i;
        }
    }
    return index;
}

int main() {

    auto images = ParseCSV("mnist_test.csv");
    
    NeuralNetwork<double> nn(784, {10}, 10);
    std:: cout << images.size() << std::endl;

    for (int i = 0; i < 100; i++) {
        double right = 0;
        double wrong = 0;
        for (const auto& image: images) {
            nn.set_input_layer(image.data);
            nn.ForwardPass();
            nn.Prop(GetGroundTruth(image.label));
            if (image.label == ReturnSelection(nn.output_layer())) {
                right++;   
            } else { wrong++; }
            // nn.PrintOutput();
        }
        // std::cout << "Accuracy: " << right / (right + wrong) << std::endl;
    }

    return 0;
}
