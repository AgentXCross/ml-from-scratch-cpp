#include "models/random_forest.hpp"

#include "preprocessing/dataset.hpp"
#include "preprocessing/train_test_split.hpp"

#include "core/metrics/accuracy.hpp"
#include "core/metrics/f1_score.hpp"

#include <iostream>
#include <cmath>

int main(void) {
    Dataset dataset = read_csv_dataset(
        "data/wisconsin_breast_cancer.csv",
        30,
        true
    );

    DatasetSplit split = train_test_split(dataset, 0.2, true);

    Matrix X_train = split.train.X;
    Matrix X_test = split.test.X;

    Matrix y_train = split.train.y;
    Matrix y_test = split.test.y;

    int num_trees = 10;
    int max_depth = 5;
    int min_samples_split = 4;
    int max_features = std::sqrt(X_train.cols());

    RandomForest model(num_trees, max_depth, min_samples_split, max_features);

    model.fit(X_train, y_train);
    std::cout << "Random Forest model fitted to training data." << "\n";

}