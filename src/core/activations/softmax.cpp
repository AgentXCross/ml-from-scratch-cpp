#include "core/activations/softmax.hpp"

#include <cmath>
#include <stdexcept>

// row-wise softmax
Matrix softmax(const Matrix &x) {
    if (x.rows() == 0 || x.cols() == 0) {
        throw std::invalid_argument("Cannot compute softmax of an empty matrix");
    }
    Matrix result(x.rows(), x.cols());

    Matrix summation_of_exp = Matrix::zeros(x.rows(), 1);

    double sum = 0.0;

    for (int i = 0; i < x.rows(); i++) {
        sum = 0.0;

        for (int j = 0; j < x.cols(); j++) {
            sum += std::exp(x.at(i, j));
        }

        summation_of_exp.at(i, 0) = sum;
    }

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            result.at(i, j) = std::exp(x.at(i, j)) / summation_of_exp.at(i, 0);
        }
    }

    return result;
}