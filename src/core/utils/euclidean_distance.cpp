#include "core/utils/euclidean_distance.hpp"

#include <cmath>
#include <stdexcept>

double euclidean_distance(
    const Tensor &a,
    const Tensor &b
) {
    if (a.empty() || b.empty()) {
        throw std::invalid_argument("Cannot compute euclidean distance of empty tensors");
    }

    if (!a.has_same_shape(b)) {
        throw std::invalid_argument("Tensors must have the same shape");
    }

    Tensor difference = a - b;

    return std::sqrt(difference.square().sum());
}