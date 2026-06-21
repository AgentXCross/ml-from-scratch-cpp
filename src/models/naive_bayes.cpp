#include "models/naive_bayes.hpp"

#include "core/utils/argmax.hpp"

#include <cmath>
#include <stdexcept>

GaussianNaiveBayes::GaussianNaiveBayes() {
    means_ = Matrix();
    variances_ = Matrix();
    priors_ = Matrix();
    num_classes_ = 0;
    fitted_ = false;
}

void GaussianNaiveBayes::fit(
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

    int n_samples = X_train.rows();
    int n_features = X_train.cols();

    num_classes_ = 0;

    for (int i = 0; i < y_train.rows(); i++) {
        int class_id = static_cast<int> (y_train.at(i, 0));

        if (class_id < 0) {
            throw std::invalid_argument("Class labels must be non-negative integers");
        }

        if (class_id + 1 > num_classes_) {
            num_classes_ = class_id + 1;
        }
    }

    Matrix class_counts(1, num_classes_);

    for (int i = 0; i < n_samples, i++) {
        int class_id = static_cast<int> (y_train.at(i, 0));
        class_counts.at(0, class_id) = class_counts.at(0, class_id) + 1;
    }

    for (int c = 0; c < class_counts.size(); c++) {
        if (class_counts.at(0, c) == 0.0) {
            throw std::runtime_error("Each class must appear at least once in y_train");
        }
    }

    means_ = Matrix(num_classes_, n_features);
    variances_ = Matrix(num_classes_, n_features);
    priors_ = Matrix(1, num_classes_);


    // priors
    for (int c = 0; c < num_classes_; c++) {
        priors_.at(0, c) = class_counts.at(0, c) / n_samples;
    }

    // means
    for (int c = 0; c < num_classes_; c++) {
        for (int j = 0; j < n_features; j++) {
            double sum_x = 0.0;

            for (int i = 0; i < n_samples; i++) {
                int class_id = static_cast<int> (y_train.at(i, 0));

                if (class_id == c) {
                    sum_x = sum_x + X_train.at(i, j);
                }
            }

            double mu = sum_x / class_counts.at(0, c);
            means_.at(c, j) = mu;
        }
    }
    
    // variances
    for (int c = 0; c < num_classes_; c++) {
        for (int j = 0; j < n_features; j++) {
            double sum_squared_diff = 0.0;
            double mu = means_.at(c, j);

            for (int i = 0; i < n_samples; i++) {
                int class_id = static_cast<int> (y_train.at(i, 0));

                if (class_id == c) {
                    double diff = X_train.at(i, j) - mu;
                    sum_squared_diff = sum_squared_diff + diff * diff;
                }
            }

            double sigma_squared = sum_squared_diff / class_counts.at(0, c);

            if (sigma_squared == 0.0) {
                sigma_squared = 1e-9;
            }

            variances_.at(c, j) = sigma_squared;
        }
    }

    fitted_ = true;
}