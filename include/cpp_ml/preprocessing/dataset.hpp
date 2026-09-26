#pragma once

#include "cpp_ml/core/tensor.hpp"

#include <string>

namespace cpp_ml {

struct Dataset {
    Tensor X;
    Tensor y;
};

// Expects numerical data
Dataset read_csv_dataset(
    const std::string &filepath,
    int label_column,
    bool has_header = true
);

}
