#pragma once

#include "core/matrix.hpp"

class GaussianNaiveBayes {
private:
    Matrix means_; // (num_classes, num_features), means for every combination of class and feature
    Matrix variances_; // (num_classes, num_features), variances for every combo of class and feature
    Matrix priors_; // (1, num_classes), priors store probability of the classes when only considering frequency in the training set
    // The prior is P(c) -> Probability of class c before knowing anything else
    // The likelihood is P(x | c) -> Probability of some features given the class
    //  Since Naive Bayes assumes all features are independent given the label
    //  P(x | c) = P(x1 | c) * P(x2 | c) * ...
    // The posterior is P(c | x) -> Probability of class c after knowing the features
    // The evidence is P(x) -> Probability of the feature values

    // By Bayes Rule, posterior = (likelihood * prior) / evidence
    // because P(c | x) = (P(x | c) * P(c)) / P(x)

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