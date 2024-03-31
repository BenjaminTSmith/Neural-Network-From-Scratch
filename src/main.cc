#include <iostream>
#include "dval.h"
#include "matrix.h"

int main() {
    Matrix<Dval> mat1(3, 3);
    mat1.SetElements({1, 0, 2, -1, 1, 1, 0, -2, 2});
    Matrix<Dval> mat2(3, 3);
    mat2.SetElements({1, 0, 2, -1, -1, 1, 0, 3, -3});

    std::cout << mat1 << std::endl;
    std::cout << mat2 << std::endl;
    std::cout << mat1 * mat2;
    return 0;
}
