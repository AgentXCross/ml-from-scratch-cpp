#pragma once

#include "core/matrix.hpp"

#include <string>

struct Dataset {
    Matrix X;
    Matrix y;
};

Dataset read_csv_dataset(
    const std::string &filepath,
    int label_column,
    bool has_header = true
);