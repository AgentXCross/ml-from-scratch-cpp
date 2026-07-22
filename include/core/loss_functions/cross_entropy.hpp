#pragma once

#include "core/tensor.hpp"

// CE = -Σ(y * log(p)) where y is the truth for the class and p is the probability of the class
double cross_entropy(
    const Tensor &y_true,
    const Tensor &y_pred
);

/* 
This returns the combined softmax + cross entropy gradient
CE = -Σ(y * log(p))
p = softmax(logits)
Derivative of softmax and cross entropy simplify to
dL/dlogits = p - y
*/
Tensor cross_entropy_gradient(
    const Tensor &y_true,
    const Tensor &y_probs
);