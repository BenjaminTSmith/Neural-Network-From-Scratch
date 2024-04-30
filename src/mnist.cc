#include <iostream>

#include "file_reader.h"
#include "layer.h"

int main() {
    auto inputs = ReadDataset("mnist_train.csv");

    Layer input_layer(10, 784);
    Layer hidden_layer(10, 1);
    Layer output_layer(1, 10);


    return 0;
}
