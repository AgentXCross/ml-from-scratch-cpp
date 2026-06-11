#include "core/utils/argmax.hpp"

#include <stdexcept>

int argmax(const Matrix &x) {
    if (x.rows() == 0 || x.cols() == 0) {
        throw std::invalid_argument("Cannot compute argmax of an empty matrix");
    }

    int max_index = 0;
    double max_value = x.at(0, 0);

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            double value = x.at(i, j);

            if (value > max_value) {
                max_value = value;
                max_index = i * x.cols() + j;
            }
        }
    }

    return max_index;
}