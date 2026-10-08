#include "cpp_ml/models/random_forest_classifier.hpp"

#include <cassert>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace cpp_ml {

RandomForestClassifier::RandomForestClassifier()
    : trees_{},
      num_trees_{10},
      max_depth_{5},
      min_samples_split_{2},
      max_features_{0},
      random_seed_{42},
      fitted_{false} {}


static std::size_t validate_num_trees(std::size_t num_trees) {
    if (num_trees == 0) {
        throw std::invalid_argument("num_trees must be positive");
    }

    return num_trees;
}


static std::size_t validate_max_depth(std::size_t max_depth) {
    if (max_depth == 0) {
        throw std::invalid_argument("max_depth must be positive");
    }

    return max_depth;
}


static std::size_t validate_min_samples_split(std::size_t min_samples_split) {
    if (min_samples_split <= 1) {
        throw std::invalid_argument("min_samples_split must be greater than 1");
    } 

    return min_samples_split;
}


static std::size_t validate_max_features(std::size_t max_features) {
    return max_features;
}


RandomForestClassifier::RandomForestClassifier(
    std::size_t num_trees,
    std::size_t max_depth,
    std::size_t min_samples_split,
    std::size_t max_features,
    unsigned int random_seed
)
    : trees_{},
      num_trees_{validate_num_trees(num_trees)},
      max_depth_{validate_max_depth(max_depth)},
      min_samples_split_{validate_min_samples_split(min_samples_split)},
      max_features_{validate_max_features(max_features)},
      random_seed_{random_seed},
      fitted_{false} {}


void RandomForestClassifier::fit(
    const Tensor &X,
    const Tensor &y
) {
    if (!X.is_matrix() || !y.is_matrix()) {
        throw std::invalid_argument("X and y must be rank-2 tensors");
    }

    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }

    if (y.rows() == 0 || y.cols() != 1) {
        throw std::invalid_argument("y must be a non-empty column vector");
    }

    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have the same number of rows");
    }

    if (max_features_ > X.cols()) {
        throw std::invalid_argument("max_features cannot be greater than the number of features");
    }

    trees_.clear();
    trees_.reserve(num_trees_);

    std::mt19937 generator{random_seed_};
    std::uniform_int_distribution<std::size_t> sample_distribution{
        0,
        X.rows() - 1
    };

    // Randomly sample the original dataset with replacement N times, where N is the number of samples in the original dataset (bootstrapping)
    for (std::size_t tree_index{0}; tree_index < num_trees_; ++tree_index) {
        std::vector<std::vector<double>> X_bootstrap_values;
        std::vector<std::vector<double>> y_bootstrap_values;

        // Reserve for performance optimization
        X_bootstrap_values.reserve(X.rows());
        y_bootstrap_values.reserve(y.rows());

        for (std::size_t i{0}; i < X.rows(); ++i) {
            std::size_t sampled_index{sample_distribution(generator)};

            std::vector<double> X_row;

            for (std::size_t j{0}; j < X.cols(); ++j) {
                X_row.push_back(X.at(sampled_index, j));
            }

            X_bootstrap_values.push_back(X_row);
            y_bootstrap_values.push_back({y.at(sampled_index, 0)});
        }

        Tensor X_bootstrap{Tensor::from_vector(X_bootstrap_values)};
        Tensor y_bootstrap{Tensor::from_vector(y_bootstrap_values)};
        assert(X_bootstrap.is_matrix());
        assert(y_bootstrap.is_matrix());
        assert(X_bootstrap.rows() == X.rows());
        assert(X_bootstrap.cols() == X.cols());
        assert(y_bootstrap.rows() == y.rows());
        assert(y_bootstrap.cols() == 1);

        std::unique_ptr<DecisionTreeClassifier> tree{std::make_unique<DecisionTreeClassifier>(
            max_depth_,
            min_samples_split_,
            max_features_,
            random_seed_ + static_cast<unsigned int>(tree_index)
        )};

        tree->fit(X_bootstrap, y_bootstrap);

        trees_.push_back(std::move(tree)); // transfer ownership
    }

    assert(trees_.size() == num_trees_);

    fitted_ = true;
}


Tensor RandomForestClassifier::predict(const Tensor &X) const {
    if (!fitted_) {
        throw std::runtime_error("RandomForestClassifier must be fitted before calling predict");
    }

    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }

    if (trees_.size() != num_trees_) {
        throw std::runtime_error("RandomForestClassifier has an invalid number of fitted trees");
    }

    Tensor predictions{{X.rows(), 1}};

    // Each individual tree predicts one label per sample
    // Each element in tree_predictions is the prediction of one tree on all samples
    // So the predictions on the ith sample is the ith row
    std::vector<Tensor> tree_predictions; 
    tree_predictions.reserve(trees_.size());

    for (std::size_t tree_index{0}; tree_index < trees_.size(); ++tree_index) {
        // predict from decision_tree_classifier.cpp returns a column vector
        tree_predictions.push_back(trees_[tree_index]->predict(X)); 
        assert(tree_predictions[tree_index].is_matrix());
        assert(tree_predictions[tree_index].rows() == X.rows());
        assert(tree_predictions[tree_index].cols() == 1);
    }

    for (std::size_t i{0}; i < X.rows(); ++i) { // ith sample
        std::map<double, std::size_t> class_counts;

        for (std::size_t tree_index{0}; tree_index < tree_predictions.size(); ++tree_index) {
            double predicted_class{tree_predictions[tree_index].at(i, 0)};
            ++class_counts[predicted_class];
        }

        double best_class{0.0};
        std::size_t best_count{0};

        for (const auto &[key, value] : class_counts) {
            double class_label{key};
            std::size_t count{value};

            if (count > best_count) {
                best_count = count;
                best_class = class_label;
            }
        }

        predictions.at(i, 0) = best_class;
    }

    assert(predictions.is_matrix());
    assert(predictions.rows() == X.rows());
    assert(predictions.cols() == 1);

    return predictions;
}


std::size_t RandomForestClassifier::num_trees() const {
    return num_trees_;
}

}
