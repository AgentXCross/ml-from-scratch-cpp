#include "models/gaussian_naive_bayes.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

GaussianNaiveBayes::GaussianNaiveBayes() 
    : means_(Tensor()),
      variances_(Tensor()),
      priors_(Tensor()),
      num_classes_(0),
      fitted_(false) {}


void GaussianNaiveBayes::fit(
    const Tensor &X_train,
    const Tensor &y_train
) {
    if (!X_train.is_matrix() || !y_train.is_matrix()) {
        throw std::invalid_argument("X_train and y_train must be rank-2 tensors");
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

    int n_samples = X_train.rows();
    int n_features = X_train.cols();

    num_classes_ = 0;

    for (int i = 0; i < y_train.rows(); i++) {
        double label = y_train.at(i, 0);
        int class_id = static_cast<int> (y_train.at(i, 0));
        
        if (label != class_id) {
            throw std::invalid_argument("Class labels must be integers");
        }

        if (class_id < 0) {
            throw std::invalid_argument("Class labels must be non-negative integers");
        }

        if (class_id + 1 > num_classes_) {
            num_classes_ = class_id + 1;
        }
    }

    Tensor class_counts({1, num_classes_});

    for (int i = 0; i < n_samples; i++) {
        int class_id = static_cast<int> (y_train.at(i, 0));
        class_counts.at(0, class_id) += 1;
    }

    for (int c = 0; c < num_classes_; c++) {
        if (class_counts.at(0, c) == 0.0) {
            throw std::runtime_error("Each class must appear at least once in y_train");
        }
    }

    means_ = Tensor({num_classes_, n_features});
    variances_ = Tensor({num_classes_, n_features});
    priors_ = Tensor({1, num_classes_});


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

    assert(means_.shape() == std::vector<int>({num_classes_, n_features}));
    assert(variances_.shape() == std::vector<int>({num_classes_, n_features}));
    assert(priors_.shape() == std::vector<int>({1, num_classes_}));

    fitted_ = true;
}

Tensor GaussianNaiveBayes::predict_probs(const Tensor &X) const {
    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (!fitted_) {
        throw std::runtime_error("GuassianNaiveBayes must be fitted before called predict_probs");
    }

    if (X.cols() != means_.cols()) {
        throw std::invalid_argument("X must have the same number of columns as the fitted data");
    }

    Tensor probabilities({X.rows(), num_classes_});

    const double pi = 3.14159265358979323846;
    const double epsilon = 1e-15;

    for (int i = 0; i < X.rows(); i++) { // loop over different samples
        // Posteriors are P(c | x)
        Tensor log_posteriors({1, num_classes_});

        for (int c = 0; c < num_classes_; c++) {
            double log_prior = std::log(priors_.at(0, c));
            double log_likelihood = 0.0;

            for (int j = 0; j < X.cols(); j++) {
                double x = X.at(i, j);
                double mu = means_.at(c, j);
                double sigma_squared = variances_.at(c, j);
                
                // log of Gaussian PDF
                double log_gaussian_density = 
                    -0.5 * std::log(2.0 * pi * sigma_squared)
                    -((x - mu) * (x - mu)) / (2.0 * sigma_squared);

                log_likelihood += log_gaussian_density;
            }

            log_posteriors.at(0, c) = log_prior + log_likelihood;
        }

        double max_log_posterior = log_posteriors.at(0, 0);

        for (int c = 1; c < num_classes_; c++) {
            if (log_posteriors.at(0, c) > max_log_posterior) {
                max_log_posterior = log_posteriors.at(0, c);
            }
        }

        double normalizing_sum = 0.0;

        for (int c = 0; c < num_classes_; c++) {
            double shifted_posterior = 
                std::exp(log_posteriors.at(0, c) - max_log_posterior);
            
            probabilities.at(i, c) = shifted_posterior;
            normalizing_sum += shifted_posterior;
        }

        assert(normalizing_sum > 0.0);

        for (int c = 0; c < num_classes_; c++) {
            probabilities.at(i, c) = probabilities.at(i, c) / normalizing_sum;
        }
    }

    assert(probabilities.is_matrix());
    assert(probabilities.rows() == X.rows());
    assert(probabilities.cols() == num_classes_);

    return probabilities;
}

Tensor GaussianNaiveBayes::predict(const Tensor &X) const {
    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    Tensor probabilities = predict_probs(X);
    Tensor predictions({X.rows(), 1});
    assert(probabilities.is_matrix());
    assert(probabilities.rows() == X.rows() && probabilities.cols() == num_classes_);

    for (int i = 0; i < probabilities.rows(); i++) {
        Tensor row = probabilities.row(i);
        int predicted_class = row.argmax();

        predictions.at(i, 0) = static_cast<double> (predicted_class);
    }

    assert(predictions.is_matrix());
    assert(predictions.rows() == X.rows());
    assert(predictions.cols() == 1);

    return predictions;
}

Tensor GaussianNaiveBayes::means() const {
    return means_;
}

Tensor GaussianNaiveBayes::variances() const {
    return variances_;
}

Tensor GaussianNaiveBayes::priors() const {
    return priors_;
}

int GaussianNaiveBayes::num_classes() const {
    return num_classes_;
}