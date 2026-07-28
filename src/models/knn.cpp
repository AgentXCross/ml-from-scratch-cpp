#include "models/knn.hpp"

#include "core/utils/euclidean_distance.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

KNN::KNN() 
    : X_train_(Tensor()),
      y_train_(Tensor()),
      k_(1),
      fitted_(false) {}


static int validate_k(int k) {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }

    return k;
}

KNN::KNN(int k) 
    : X_train_(Tensor()),
      y_train_(Tensor()),
      k_(validate_k(k)),
      fitted_(false) {}

void KNN::fit(
    const Tensor &X_train,
    const Tensor &y_train
) {
    if (!X_train.is_matrix() || !y_train.is_matrix()) {
        throw std::invalid_argument("X_train and y_train must be rank-2 tensor");
    }
    if (X_train.rows() == 0 || X_train.cols() == 0) {
        throw std::invalid_argument("X_train cannot be empty");
    }

    if (y_train.rows() == 0 || y_train.cols() == 0) {
        throw std::invalid_argument("y_train cannot be empty");
    }

    if (X_train.rows() != y_train.rows()) {
        throw std::invalid_argument("X_train and y_train must have the same number of rows");
    }

    if (y_train.cols() != 1) {
        throw std::invalid_argument("y_train must have exactly one column");
    }

    if (k_ > X_train.rows()) {
        throw std::invalid_argument("k cannot be larger than number of training samples");
    }

    X_train_ = X_train;
    y_train_ = y_train;
    fitted_ = true;
}

Tensor KNN::predict(const Tensor &X) const {
    if (!fitted_) {
        throw std::runtime_error("KNN must be fitted before calling predict");
    }

    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of columns as X_train");
    }

    int num_classes = 0; // go through data and determine how many classes there are

    for (int i = 0; i < y_train_.rows(); i++) {
        int label = static_cast<int> (y_train_.at(i, 0));

        if (label < 0) {
            throw std::invalid_argument("KNN labels must be non-negative");
        }

        if (label + 1 > num_classes) {
            num_classes = label + 1;
        }
    }

    Tensor predictions({X.rows(), 1});

    for (int i = 0; i < X.rows(); i++) { // loop through every sample
        std::vector<std::pair<double, double>> distances_and_labels;

        Tensor sample = X.row(i);
        assert(sample.rows() == 1);

        for (int train_i = 0; train_i < X_train_.rows(); train_i++) {
            Tensor train_sample = X_train_.row(train_i);
            assert(train_sample.rows() == 1);

            double distance = euclidean_distance(sample, train_sample);
            double label = y_train_.at(train_i, 0);

            distances_and_labels.push_back({distance, label});
        }

        std::sort(
            distances_and_labels.begin(),
            distances_and_labels.end(),
            [](const std::pair<double, double> &a, const std::pair<double, double> &b) {
                return a.first < b.first;
            }
        );

        Tensor class_counts({1, num_classes});

        for (int neighbor = 0; neighbor < k_; neighbor++) {
            int label = static_cast<int> (distances_and_labels[neighbor].second);

            class_counts.at(0, label) = class_counts.at(0, label) + 1.0;
        }

        int predicted_label = class_counts.argmax();

        predictions.at(i, 0) = static_cast<double> (predicted_label);
    }

    return predictions;
}

int KNN::k() const {
    return k_;
}