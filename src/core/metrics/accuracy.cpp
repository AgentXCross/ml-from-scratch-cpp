#include "core/metrics/accuracy.hpp"

#include <stdexcept>


double accuracy_score(
    const Tensor &y_true,
    const Tensor &y_pred
) {
    if (y_true.empty() || y_pred.empty()) {
        throw std::invalid_argument("y_true and y_pred cannot be empty");
    }

    if (!y_true.has_same_shape(y_pred)) {
        throw std::invalid_argument("y_true and y_pred must have the same shape");
    }

    int num_elements = y_true.size();
    int num_correct = 0;

    for (int i = 0; i < y_true.size(); i++) {
        if (y_true.at_flat(i) == y_pred.at_flat(i)) {
            num_correct++;
        }
    }

    return static_cast<double> (num_correct) / num_elements;
}
