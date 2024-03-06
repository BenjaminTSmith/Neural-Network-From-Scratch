#include <iostream>
#include "Node.h"

int main() {

    /*Eigen::MatrixXd input(1, 3);
    input << 7, 8, 9;
    std::cout << input << std::endl;
    Eigen::MatrixXd ground_truth(1, 3);
    ground_truth << 3, -2, -1;

    nn::Layer hidden_layer(5, 3);
    nn::Layer output_layer(1, 5);

    hidden_layer.ForwardPass(input);
    output_layer.ForwardPass(hidden_layer);*/

    Eigen::MatrixXd test = Eigen::MatrixXd::Ones(3, 3);
    nn::Node one(test);
    nn::Node two(test);
    nn::Node three(test);

    nn::Node six = one + two + three;
    six *= 2;
    six -= 1;
    six.delta_ = Eigen::MatrixXd::Ones(3, 3);
    std::cout << six.value_ << std::endl;
    six();
    std::cout << six.value_ << std::endl;

    return 0;
}
