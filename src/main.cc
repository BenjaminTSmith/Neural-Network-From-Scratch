#include <iostream>
#include "dval.h"
#include "matrix.h"

int main() {

    Dval x(2.0);
    Dval y(3.0);
    Dval z = x * y;
    std::cout << z << std::endl;

    Matrix<Dval> mat1(3, 3);
    mat1.SetElements({1, 0, 0, 0, 1, 0, 0, 0, 1});
    Matrix<Dval> mat2(3, 3);
    mat2.SetElements({1, 0, 0, 0, 1, 0, 0, 0, 1});
    auto mat3 = mat1 + mat2;

    std::cout << mat1 << std::endl << mat2 << std::endl;
    std::cout << mat3;
    return 0;
}
