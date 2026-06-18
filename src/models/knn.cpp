#include "models/knn.hpp"

#include "core/utils/argmax.hpp"
#include "core/utils/euclidean_distance.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

KNN::KNN() {
    X_train_ = Matrix();
    y_train_ = Matrix();
    k_ = 1;
    fitted_ = false;
}

KNN::KNN(int k) {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }

    X_train_ = Matrix();
    y_train_ = Matrix();
    k_ = k;
    fitted_ = false;
}

void KNN::fit(
    const Matrix &X_train,
    const Matrix &y_train
) {
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
        throw std::invalid_argument("k cannot be larger than number of trianing samples");
    }

    X_train_ = X_train;
    y_train_ = y_train;
    fitted_ = true;
}

Matrix KNN::predict(const Matrix &X) const {
    if (!fitted_) {
        throw std::runtime_error("KNN must be fitted before calling predict");
    }

    if (X.cols() != X_train_.cols()) {
        throw std::invalid_argument("X must have the same number of columns at X_train");
    }

    int num_classes = 0;

    for (int i = 0; i < y_train_.rows(); i++) {
        int label = static_cast<int> (y_train_.at(i, 0));

        if (label < 0) {
            throw std::invalid_argument("KNN labels must be non-negative");
        }

        if (label + 1 > num_classes) {
            num_classes = label + 1;
        }
    }

    Matrix predictions(X.rows(), 1);

    for (int i = 0; i < X.rows(); i++) {
        std::vector<std::pair<double, double>> distances_and_labels;

        Matrix sample = X.row(i);

        for (int train_i = 0; train_i < X_train_.rows(); train_i++) {
            Matrix train_sample = X_train_.row()
        }
    }
}