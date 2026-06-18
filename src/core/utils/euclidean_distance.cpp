#include "core/utils/euclidean_distance.hpp"

#include <cmath>
#include <stdexcept>

double euclidean_distance(
    const Matrix &a,
    const Matrix &b
) {
    if (a.rows() != 1 || b.rows() != 1) {
        throw std::invalid_argument("a and b must be two row matrices");
    }

    if (a.cols() != b.cols()) {
        throw std::invalid_argument("a and b must have the same number of columns");
    }

    double squared_distance = 0.0;

    for (int j = 0; j < a.cols(); j++) {
        double difference = a.at(0, j) - b.at(0, j);

        squared_distance = squared_distance + difference * difference;
    }

    return std::sqrt(squared_distance);
}