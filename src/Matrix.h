#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <vector>


static std::vector<double> OneHotEncode(int num, int size) {
    std::vector<double> one_hot_vector;
    one_hot_vector.resize(size);
    std::fill(one_hot_vector.begin(), one_hot_vector.end(), 0);
    one_hot_vector[num] = 1;
    return one_hot_vector;
}

static void Clip(std::vector<double>& input, double min, double max) {
    for (auto& in: input) {
        in = in < min ? min : in;
        in = in > max ? max : in;
    }
}


#endif // !MATRIX_H
