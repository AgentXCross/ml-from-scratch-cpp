#include "preprocessing/one_hot_encode.hpp"

#include <stdexcept>

Matrix one_hot_encode(
    const Matrix &y,
    int num_classes
) {
    if (y.cols() != 1) {
        throw std::invalid_argument("y must have exactly one column");
    }

    if (num_classes <= 1) {
        throw std::invalid_argument("num_classes must be at least 2");
    }

    Matrix result(y.rows(), num_classes);

    for (int i = 0; i < y.rows(); i++) {
        int class_id = static_cast<int> (y.at(i, 0));

        if (class_id < 0 || class_id >= num_classes) {
            throw std::invalid_argument("class label is out of range");
        }

        result.at(i, class_id) = 1.0;
    }

    return result;
}