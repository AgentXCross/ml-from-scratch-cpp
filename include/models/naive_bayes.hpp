#pragma once

#include "core/matrix.hpp"

class GaussianNaiveBayes {
private:
    Matrix means_; // (num_classes, num_features), means for every combination of class and feature
    Matrix variances_; // (num_classes, num_features), variances for every combo of class and feature
    Matrix priors_; // (1, num_classes), priors store probability of the classes when only considering frequency in the training set

    int num_classes_;
    bool fitted_;

public:
    GaussianNaiveBayes();

    void fit(
        const Matrix &X_train,
        const Matrix &y_train
    );

    Matrix predict(const Matrix &X) const;
    Matrix predict_probs(const Matrix &X) const;

    Matrix means() const;
    Matrix variances() const;
    Matrix priors() const;

    int num_classes() const;
};