#ifndef MATRIX_H
#define MATRIX_H

#include <stdexcept>
#include <iostream>
#include <vector>

template <typename T>
struct Matrix {
    int row_count_;
    int col_count_;
    std::vector<T> elements_;

    Matrix(int row_count, int col_count) 
        : row_count_(row_count),
          col_count_(col_count) {}

    Matrix(const Matrix& other) = default;

    T operator[](size_t idx) const { return elements_[idx]; }

    void SetElements(const std::vector<T> elements) {
        if (elements.size() != row_count_ * col_count_)
            throw std::range_error("Size of elements does not match size of matrix!");
        elements_ = elements; 
    }

    // dot product
    Matrix operator*(const Matrix& other) {
        if (col_count_ != other.row_count_)
            throw std::range_error("Matrices aren't the correct sizes!");

        Matrix ret(row_count_, other.col_count_);
        std::vector<T> new_elements;
        for (size_t i = 0; i < row_count_; i++) {
            for (size_t j = 0; j < other.col_count_; j++) {
                T elem = 0;
                for (size_t k = 0; k < other.row_count_; k++) {
                    elem += elements_[i * col_count_ + k] * other[k * other.col_count_ + j];
                }
                new_elements.push_back(elem);
            }
        }

        ret.SetElements(new_elements);
        return ret;
    }

    // coeff wise addition
    Matrix operator+(const Matrix& other) {
        if (other.row_count_ != row_count_ or other.col_count_ != col_count_) 
            throw std::range_error("Matrix sizes don't match!");
        
        Matrix ret(row_count_, col_count_);
        std::vector<T> new_elements;
        for (size_t i = 0; i < elements_.size(); i++) 
            new_elements.push_back(elements_[i] + other.elements_[i]);

        ret.SetElements(new_elements);
        return ret;
    }

};

template <typename T>
std::ostream& operator<<(std::ostream& oss, const Matrix<T>& mat) {
    for (size_t i = 0; i < mat.row_count_; i++) {
        for (size_t j = 0; j < mat.col_count_; j++) {
            std::cout << mat[i * mat.col_count_ + j] << " "; 
        }
        std::cout << std::endl;
    }
    return oss;
}

#endif // !MATRIX_H
