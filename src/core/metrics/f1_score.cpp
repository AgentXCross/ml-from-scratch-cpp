#include "core/metrics/f1_score.hpp"

#include "core/metrics/precision.hpp"
#include "core/metrics/recall.hpp"


double f1_score(
    const Tensor &y_true,
    const Tensor &y_pred,
    double positive_class
) {
    double precision = precision_score(y_true, y_pred, positive_class);
    double recall = recall_score(y_true, y_pred, positive_class);

    if (precision + recall == 0.0) {
        return 0.0;
    }

    return (2.0 * precision * recall) / (precision + recall);
}

