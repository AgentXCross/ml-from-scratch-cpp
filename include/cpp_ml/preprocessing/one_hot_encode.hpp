#pragma once

#include "cpp_ml/core/tensor.hpp"

namespace cpp_ml {

/*
one_hot_encode takes a column vector of class labels and 
turns each label into a one-hot row vector.
*/
Tensor one_hot_encode(
    const Tensor &y,
    int num_classes
);

}
