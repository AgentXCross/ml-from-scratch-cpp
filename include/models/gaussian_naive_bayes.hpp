#pragma once

#include "core/tensor.hpp"

class GaussianNaiveBayes {
private:
    Tensor means_; // (num_classes, num_features), means for every combination of class and feature
    Tensor variances_; // (num_classes, num_features), variances for every combo of class and feature
    Tensor priors_; // (1, num_classes), priors store probability of the classes when only considering frequency in the training set

    // The prior is P(c) -> Probability of class c before knowing anything else
    // We get the prior by the proportion of training samples that are of class c

    // The likelihood is P(x | c) -> Probability of some features given the class
    // Since Naive Bayes assumes all features are independent given the label
    // P(x | c) = P(x1 | c) * P(x2 | c) * ...
    // For Guassian Naive Bayes, we assume all the features follow a normal distribution when calculating the probabilities

    // The evidence is P(x) -> Probability of the feature values
    // P(x) = Σ P(x | c) * P(c) for all classes c, assuming each feature follows a normal distribution

    // The posterior is P(c | x) -> Probability of class c after knowing the features
    // The posterior is what we need to calculate for every class

    // By Bayes Rule, posterior = (likelihood * prior) / evidence
    // because P(c | x) = (P(x | c) * P(c)) / P(x)

    int num_classes_;
    bool fitted_;

public:
    GaussianNaiveBayes();

    void fit(
        const Tensor &X_train,
        const Tensor &y_train
    );

    Tensor predict(const Tensor &X) const;
    Tensor predict_probs(const Tensor &X) const;

    Tensor means() const;
    Tensor variances() const;
    Tensor priors() const;

    int num_classes() const;
};