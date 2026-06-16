#include "core/activations/tanh.hpp"

#include <cmath>

Matrix tanh(const Matrix &x) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            double value = x.at(i, j);

            result.at(i, j) = (std::exp(value) - std::exp(-value)) / 
                                (std::exp(value) + std::exp(-value));
        }
    }

    return result;
}

Matrix tanh_gradient(const Matrix &x) {
    Matrix y = tanh(x);

    Matrix ones = Matrix::ones(x.rows(), x.cols());

    return ones - y.elementwise_multiply(y);
}