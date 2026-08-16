#include "preprocessing/train_test_split.hpp"

#include <algorithm>
#include <cassert>
#include <random>
#include <stdexcept>
#include <vector>

DatasetSplit train_test_split(
    const Dataset &dataset,
    double test_size,
    bool shuffle
) {
    if (!dataset.X.is_matrix() || !dataset.y.is_matrix()) {
        throw std::invalid_argument("X and y must be rank-2 tensors");
    }

    if (dataset.X.rows() == 0 || dataset.y.rows() == 0) {
        throw std::invalid_argument("Cannot split an empty dataset");
    }

    if (dataset.X.rows() != dataset.y.rows()) {
        throw std::invalid_argument("X and y must have the same number of rows");
    }

    if (dataset.y.cols() != 1) {
        throw std::invalid_argument("y must have exactly one column");
    }

    if (test_size <= 0.0 || test_size >= 1.0) {
        throw std::invalid_argument("test_size must be between 0.0 and 1.0 exclusive");
    }

    int num_samples = dataset.X.rows();
    int num_test = static_cast<int> (num_samples * test_size);
    int num_train = num_samples - num_test;

    if (num_test == 0 || num_train == 0) {
        throw std::invalid_argument("test_size creates an empty train or test split");
    }

    std::vector<int> indices;

    for(int i = 0; i < num_samples; i++) {
        indices.push_back(i);
    }

    if (shuffle) {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::shuffle(indices.begin(), indices.end(), gen);
    }

    std::vector<std::vector<double>> X_train;
    std::vector<std::vector<double>> y_train;
    std::vector<std::vector<double>> X_test;
    std::vector<std::vector<double>> y_test;

    for (int i = 0; i < num_samples; i++) {
        int original_index = indices[i];

        std::vector<double> X_row;
        std::vector<double> y_row;

        for (int j = 0; j < dataset.X.cols(); j++) {
            X_row.push_back(dataset.X.at(original_index, j));
        }

        for (int j = 0; j < dataset.y.cols(); j++) {
            y_row.push_back(dataset.y.at(original_index, j));
        }

        if (i < num_train) {
            X_train.push_back(X_row);
            y_train.push_back(y_row);
        } else {
            X_test.push_back(X_row);
            y_test.push_back(y_row);
        }
    }

    DatasetSplit split;

    split.train.X = Tensor::from_vector(X_train);
    split.train.y = Tensor::from_vector(y_train);
    split.test.X = Tensor::from_vector(X_test);
    split.test.y = Tensor::from_vector(y_test);

    assert(split.train.X.is_matrix());
    assert(split.train.y.is_matrix());
    assert(split.test.X.is_matrix());
    assert(split.test.y.is_matrix());
    assert(split.train.X.rows() == num_train);
    assert(split.train.y.rows() == num_train);
    assert(split.test.X.rows() == num_test);
    assert(split.test.y.rows() == num_test);
    assert(split.train.X.cols() == dataset.X.cols());
    assert(split.test.X.cols() == dataset.X.cols());
    assert(split.train.y.cols() == 1);
    assert(split.test.y.cols() == 1);

    return split;
}
