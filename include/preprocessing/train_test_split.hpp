#pragma once

#include "preprocessing/dataset.hpp"

struct DatasetSplit {
    Dataset train;
    Dataset test;
};

DatasetSplit train_test_split(
    const Dataset &dataset,
    double test_size = 0.2,
    bool shuffle = true
);