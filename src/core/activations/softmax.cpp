#include "cpp_ml/core/activations/softmax.hpp"

#include <cmath>
#include <stdexcept>

namespace cpp_ml {

// row-wise softmax for rank-2 Tensors only
Tensor softmax(const Tensor &x) {
    if (!x.is_matrix()) {
        throw std::invalid_argument("softmax expects a rank-2 tensor");
    }
    Tensor result{x.shape()};

    for (std::size_t i{0}; i < x.rows(); ++i) {
        double sum{0.0};

        for (std::size_t j{0}; j < x.cols(); ++j) {
            sum += std::exp(x.at(i, j));
        }

        for (std::size_t j{0}; j < x.cols(); ++j) {
            result.at(i, j) = std::exp(x.at(i, j)) / sum;
        }
    }

    return result;
}

}
