#include "knn.hpp"

#include "core/metrics/accuracy.hpp"
#include "preprocessing/dataset.hpp"
#include "preprocessing/standard_scaler.hpp"
#include "preprocessing/train_test_split.hpp"

#include <iostream>

int main(void) {
    Dataset dataset = read_csv_dataset(
        "data/digits.csv",
        64,
        true
    );

    DatasetSplit split = train_test_split(dataset, 0.2, true);

    
}