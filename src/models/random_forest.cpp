#include "models/random_forest.hpp"

#include <map>
#include <random>
#include <stdexcept>
#include <vector>

RandomForest::RandomForest() {
    num_trees_ = 10;
    max_depth_ = 5;
    min_samples_split_ = 2;
    max_features_ = 0;

    random_seed_ = 42;
    fitted_ = false;
}


RandomForest::RandomForest(
    int num_trees,
    int max_depth,
    int min_samples_split,
    int max_features,
    unsigned int random_seed
) {
    if (num_trees <= 0) {
        throw std::invalid_argument("num_trees must be positive");
    }

    if (max_depth <= 0) {
        throw std::invalid_argument("max_depth must be positive");
    }

    if (min_samples_split <= 1) {
        throw std::invalid_argument("min_samples_split must be greater than 1");
    } 

    if (max_features <= 0) {
        throw std::invalid_argument("max_features must be positive");
    }

    num_trees_ = num_trees;
    max_depth_ = max_depth;
    min_samples_split_ = min_samples_split;
    max_features_ = max_features;

    random_seed_ = random_seed;
    fitted_ = false;
}


void RandomForest::fit(
    const Matrix &X,
    const Matrix &y
) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }

    if (y.rows() == 0 || y.cols() != 1) {
        throw std::invalid_argument("y must be a non-empty column vector");
    }

    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have the same number of rows");
    }

    if (max_features_ <= 0) {
        throw std::invalid_argument("max_features must be positive");
    }

    if (max_features_ > X.cols()) {
        throw std::invalid_argument("max_features cannot be greater than the number of features");
    }

    trees_.clear();
    trees_.reserve(num_trees_);

    std::mt19937 generator(random_seed_);
    std::uniform_int_distribution<int> sample_distribution(
        0,
        X.rows() - 1
    );

    // Randomly sample the original dataset with replacement N times, where N is the number of samples in the original dataset (bootstrapping)
    for (int tree_index = 0; tree_index < num_trees_; tree_index++) {
        std::vector<std::vector<double>> X_bootstrap_values;
        std::vector<std::vector<double>> y_bootstrap_values;

        // Reserve for performance optimization
        X_bootstrap_values.reserve(X.rows());
        y_bootstrap_values.reserve(y.rows());

        for (int i = 0; i < X.rows(); i++) {
            int sampled_index = sample_distribution(generator);

            std::vector<double> X_row;

            for (int j = 0; j < X.cols(); j++) {
                X_row.push_back(X.at(sampled_index, j));
            }

            X_bootstrap_values.push_back(X_row);
            y_bootstrap_values.push_back({y.at(sampled_index, 0)});
        }

        Matrix X_bootstrap = Matrix::from_vector(X_bootstrap_values);
        Matrix y_bootstrap = Matrix::from_vector(y_bootstrap_values);

        std::unique_ptr<DecisionTree> tree = std::make_unique<DecisionTree> (
            max_depth_,
            min_samples_split_
        );

        tree->fit(X_bootstrap, y_bootstrap);

        trees_.push_back(std::move(tree)); // transfer ownership
    }

    fitted_ = true;
}


Matrix RandomForest::predict(const Matrix &X) const {
    if (!fitted_) {
        throw std::runtime_error("RandomForest must be fitted before calling predict");
    }
}
