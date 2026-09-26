#include "cpp_ml/models/random_forest_classifier.hpp"

#include "cpp_ml/preprocessing/dataset.hpp"
#include "cpp_ml/preprocessing/train_test_split.hpp"

#include "cpp_ml/core/metrics/accuracy.hpp"
#include "cpp_ml/core/metrics/f1_score.hpp"

#include <iostream>
#include <cmath>

int main(void) {
    cpp_ml::Dataset dataset = cpp_ml::read_csv_dataset(
        "data/wisconsin_breast_cancer.csv",
        30,
        true
    );

    cpp_ml::DatasetSplit split = cpp_ml::train_test_split(dataset, 0.2, true);

    cpp_ml::Tensor X_train = split.train.X;
    cpp_ml::Tensor X_test = split.test.X;

    cpp_ml::Tensor y_train = split.train.y;
    cpp_ml::Tensor y_test = split.test.y;

    int num_trees = 10;
    int max_depth = 5;
    int min_samples_split = 4;
    int max_features = static_cast<double> (std::sqrt(X_train.cols()));

    cpp_ml::RandomForestClassifier model(num_trees, max_depth, min_samples_split, max_features);

    model.fit(X_train, y_train);
    std::cout << "Random Forest model fitted to training data." << "\n";

    cpp_ml::Tensor train_predictions = model.predict(X_train);
    cpp_ml::Tensor test_predictions = model.predict(X_test);

    double train_accuracy = accuracy_score(y_train, train_predictions);
    double test_accuracy = accuracy_score(y_test, test_predictions);

    double train_f1 = f1_score(y_train, train_predictions);
    double test_f1 = f1_score(y_test, test_predictions);

    std::cout << "Train Accuracy: " << train_accuracy << " || " <<
        "Test Accuracy: " << test_accuracy << "\n";
    
    std::cout << "Train F1: " << train_f1 << " || " << 
        "Test F1: " << test_f1 << "\n";
}
