#include "core/activations/sigmoid.hpp"

#include <cmath>

Matrix sigmoid(const Matrix &x) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            result.at(i, j) = 1.0 / (1.0 + std::exp(-x.at(i, j)));
        }
    }

    return result;
}

Matrix sigmoid_gradient(const Matrix &x) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            double sig = 1.0 / (1.0 + std::exp(-x.at(i, j)));

            result.at(i, j) = sig * (1 - sig);
        }
    }

    return result;
}