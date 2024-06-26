#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>

template <typename T> struct Matrix {
    int row_count_;
    int col_count_;
    std::vector<T> elements_;

    Matrix(int row_count, int col_count)
        : row_count_(row_count), col_count_(col_count),
          elements_(row_count * col_count) {}

    Matrix() : row_count_(0), col_count_(0) {}

    Matrix(const Matrix& other) = default;

    T& operator[](size_t idx) { return elements_[idx]; }
    const T& operator[](size_t idx) const { return elements_[idx]; }

    size_t size() const { return elements_.size(); }

    void SetElements(const std::vector<T>& elements) {
        // elements are in row order. i.e. a matrix that looks like:
        // 0 1 -1
        // 1 2 -3
        // 3 1 0
        // would be ordered in the vector
        // like this: { 0, 1, -1, 1, 2, -3, 3, 1, 0 }

        if (elements.size() != row_count_ * col_count_)
            throw std::range_error(
                "Size of elements does not match size of matrix!");
        elements_ = elements;
    }

    // dot product
    Matrix operator*(const Matrix& other) const {
        if (col_count_ != other.row_count_)
            throw std::range_error("Matrices aren't the correct sizes!");

        Matrix ret(row_count_, other.col_count_);
        std::vector<T> new_elements;
        for (size_t i = 0; i < row_count_; i++) {
            for (size_t j = 0; j < other.col_count_; j++) {
                T elem = 0;
                for (size_t k = 0; k < other.row_count_; k++) {
                    elem += elements_[i * col_count_ + k] *
                            other[k * other.col_count_ + j];
                }
                new_elements.push_back(elem);
            }
        }

        ret.SetElements(new_elements);
        return ret;
    }

    // coeff wise addition
    Matrix operator+(const Matrix& other) const {
        if (other.row_count_ != row_count_ or other.col_count_ != col_count_)
            throw std::range_error("Matrix sizes don't match!");

        Matrix ret(row_count_, col_count_);
        std::vector<T> new_elements;
        for (size_t i = 0; i < elements_.size(); i++)
            new_elements.push_back(elements_[i] + other.elements_[i]);

        ret.SetElements(new_elements);
        return ret;
    }

    Matrix operator-(const Matrix& other) const {
        if (other.row_count_ != row_count_ or other.col_count_ != col_count_)
            throw std::range_error("Matrix sizes don't match!");

        Matrix ret(row_count_, col_count_);
        std::vector<T> new_elements;
        for (size_t i = 0; i < elements_.size(); i++)
            new_elements.push_back(elements_[i] - other.elements_[i]);

        ret.SetElements(new_elements);
        return ret;
    }

    // coeff wise scalar operations
    Matrix operator+(const T& scalar) const {
        std::vector<T> new_elements;
        for (const auto& element_ : elements_)
            new_elements.push_back(element_ + scalar);
        Matrix ret(row_count_, col_count_);
        ret.SetElements(new_elements);
        return ret;
    }

    Matrix operator*(const T& scalar) const {
        std::vector<T> new_elements;
        for (const auto& element_ : elements_)
            new_elements.push_back(element_ * scalar);
        Matrix ret(row_count_, col_count_);
        ret.SetElements(new_elements);
        return ret;
    }

    Matrix ColwiseAverage() const {
        Matrix avg(1, col_count_);
        std::vector<T> new_elements;
        for (size_t i = 0; i < col_count_; i++) {
            T sum = 0;
            for (size_t j = 0; j < row_count_; j++) {
                sum += elements_[j * col_count_ + i];
            }
            new_elements.push_back(sum / row_count_);
        }
        avg.SetElements(new_elements);
        return avg;
    }

    Matrix RowwiseAverage() const {
        Matrix avg(row_count_, 1);
        std::vector<T> new_elements;
        for (size_t i = 0; i < row_count_; i++) {
            T sum = 0;
            for (size_t j = 0; j < col_count_; j++) {
                sum += elements_[i * col_count_ + j];
            }
            new_elements.push_back(sum / row_count_);
        }
        avg.SetElements(new_elements);
        return avg;
    }

    Matrix Square() const {
        Matrix ret(row_count_, col_count_);
        std::vector<T> new_elements;
        for (const auto& element_ : elements_)
            new_elements.push_back(element_ * element_);
        ret.SetElements(new_elements);
        return ret;
    }

    T Average() {
        T average = 0;
        for (const auto& element_ : elements_) {
            average += element_;
        }
        average /= elements_.size();
        return average;
    }

    Matrix Transpose() const {
        Matrix ret(col_count_, row_count_);
        std::vector<T> new_elements;
        for (size_t i = 0; i < col_count_; i++) {
            for (size_t j = 0; j < row_count_; j++) {
                new_elements.push_back(elements_[j * col_count_ + i]);
            }
        }
        ret.SetElements(new_elements);
        return ret;
    }

    Matrix Max(const T& scalar = 0) const {
        Matrix ret(row_count_, col_count_);
        std::vector<T> new_elements;
        for (const auto& element_ : elements_) {
            if (element_ > scalar)
                new_elements.push_back(element_);
            else
                new_elements.push_back(scalar);
        }
        ret.SetElements(new_elements);
        return ret;
    }

    Matrix Min(const T& scalar = 0) const {
        Matrix ret(row_count_, col_count_);
        std::vector<T> new_elements;
        for (const auto& element_ : elements_) {
            if (element_ < scalar)
                new_elements.push_back(element_);
            else
                new_elements.push_back(scalar);
        }
        ret.SetElements(new_elements);
        return ret;
    }

    Matrix SoftMax() const {
        Matrix result(row_count_, col_count_);
        std::vector<T> maxes;
        std::vector<T> sums;
        std::vector<T> soft_max_elements;

        for (int i = 0; i < col_count_; i++) {
            maxes.push_back(elements_[i]);
        }

        for (int i = 0; i < col_count_; i++) {
            for (int j = 1; j < row_count_; j++) {
                if (elements_[j * col_count_ + i] > maxes[i])
                    maxes[i] = elements_[j * col_count_ + i];
            }
        }

        for (int i = 0; i < col_count_; i++) {
            sums.push_back(std::exp(elements_[i] - maxes[i]));
        }

        for (int i = 0; i < col_count_; i++) {
            for (int j = 1; j < row_count_; j++) {
                sums[i] += std::exp(elements_[j * col_count_ + i] - maxes[i]);
            }
        }


        for (int i = 0; i < size(); i++) {
            soft_max_elements.push_back(
                std::exp(elements_[i] - maxes[i % col_count_]) /
                sums[i % col_count_]
            );
        }

        result.SetElements(soft_max_elements);
        return result;
    }
};

template <typename T>
static std::ostream& operator<<(std::ostream& oss, const Matrix<T> mat) {
    for (size_t i = 0; i < mat.row_count_ - 1; i++) {
        for (size_t j = 0; j < mat.col_count_; j++) {
            oss << mat[i * mat.col_count_ + j] << ' ';
        }
        oss << std::endl;
    }
    for (size_t j = 0; j < mat.col_count_; j++) {
        oss << mat[(mat.row_count_ - 1) * mat.col_count_ + j] << ' ';
    }
    return oss;
}

;
#endif // !MATRIX_H
