#include "cpp_ml/preprocessing/one_hot_encode.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace cpp_ml {

Tensor one_hot_encode(
    const Tensor &y,
    std::size_t num_classes
) {
    if (!y.is_matrix()) {
        throw std::invalid_argument("y must be a rank-2 tensor");
    }

    if (y.rows() == 0 || y.cols() == 0) {
        throw std::invalid_argument("y cannot be empty");
    }

    if (y.cols() != 1) {
        throw std::invalid_argument("y must have exactly one column");
    }

    if (num_classes <= 1) {
        throw std::invalid_argument("num_classes must be at least 2");
    }

    Tensor result{{y.rows(), num_classes}};

    for (std::size_t i{0}; i < y.rows(); ++i) {
        double label{y.at(i, 0)};

        if (label < 0.0 || std::floor(label) != label) {
            throw std::invalid_argument("class labels must be integers");
        }

        std::size_t class_id{static_cast<std::size_t>(label)};

        if (class_id >= num_classes) {
            throw std::invalid_argument("class label is out of range");
        }

        result.at(i, class_id) = 1.0;
    }

    assert(result.is_matrix());
    assert(result.rows() == y.rows());
    assert(result.cols() == num_classes);

    return result;
}

}
