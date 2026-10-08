#include "cpp_ml/core/metrics/precision.hpp"

#include <stdexcept>

namespace cpp_ml {

double precision_score(
    const Tensor &y_true,
    const Tensor &y_pred,
    double positive_class
) {
    if (y_true.empty() || y_pred.empty()) {
        throw std::invalid_argument("y_true and y_pred cannot be empty");
    }

    if (!y_true.has_same_shape(y_pred)) {
        throw std::invalid_argument("y_true and y_pred must have the same shape");
    }

    std::size_t tp{0};
    std::size_t fp{0};

    for (std::size_t i{0}; i < y_true.size(); ++i) {
        double truth{y_true.at_flat(i)};
        double pred{y_pred.at_flat(i)};

        if (pred == positive_class) {
            if (truth == positive_class) {
                ++tp;
            } else if (truth != positive_class) {
                ++fp;
            }
        }
    }

    if (tp + fp == 0) {
        return 0.0;
    }

    return static_cast<double>(tp) / static_cast<double>(tp + fp);
}

}
