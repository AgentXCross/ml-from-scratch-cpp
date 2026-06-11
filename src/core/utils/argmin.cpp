#include "core/utils/argmin.hpp"

#include <stdexcept>

int argmin(const Matrix &x) {
    if (x.rows() == 0 || x.cols() == 0) {
        throw std::invalid_argument("Cannot compute argmin of an empty matrix");
    }

    int min_index = 0;
    double min_value = x.at(0, 0);

    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            double value = x.at(i, j);

            if (value < min_value) {
                min_value = value;
                min_index = i * x.cols() + j;
            }
        }
    }

    return min_index;
}