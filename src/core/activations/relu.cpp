#include "core/activations/relu.hpp"

#include <algorithm>

Matrix relu(const Matrix &x) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            result.at(i, j) = std::max(0.0, x.at(i, j));
        }
    }

    return result;
}

Matrix relu_gradient(const Matrix &x) {
    Matrix result(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            result.at(i, j) = (x.at(i, j) > 0.0) ? 1.0 : 0.0;
        }
    }

    return result;
}