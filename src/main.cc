#include <iostream>
#include "dval.h"
#include "matrix.h"
#include "neuron.h"

int main() {

    Matrix<Dval> mat1(2, 3);
    mat1.SetElements({ -2, 1, 4, 5, -9, 0 });
    std::cout << mat1 << std::endl << std::endl << mat1.Transpose().Max(0) << std::endl << std::endl;
    std::cout << mat1.ColwiseAverage() << std::endl << std::endl;
    std::cout << mat1 * -1 << std::endl;
    mat1 = mat1 * -1;

    return 0;
}
