#include "core/utils/threshold.hpp"

Tensor threshold(
    const Tensor &x, 
    double cutoff,
    double upper,
    double lower
) {
    if (x.empty()) {
        throw std::invalid_argument("Cannot threshold an empty tensor");
    }

    Tensor result(x.shape());

    for (int i = 0; i < x.size(); i++) {
        if (x.at_flat(i) >= cutoff) {
            result.at_flat(i) = upper;
        } else {
            result.at_flat(i) = lower;
        }
    }

    return result;
}