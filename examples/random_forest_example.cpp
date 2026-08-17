#include "models/random_forest_classifier.hpp"

#include "preprocessing/dataset.hpp"
#include "preprocessing/train_test_split.hpp"

#include "core/metrics/accuracy.hpp"
#include "core/metrics/f1_score.hpp"

#include <iostream>
#include <cmath>

int main(void) {
    Dataset dataset = read_csv_dataset(
        "data/wisconsin_breast_cancer.csv",
        30,
        true
    );

    DatasetSplit split = train_test_split(dataset, 0.2, true);

    Tensor X_train = split.train.X;
    Tensor X_test = split.test.X;

    Tensor y_train = split.train.y;
    Tensor y_test = split.test.y;

    int num_trees = 10;
    int max_depth = 5;
    int min_samples_split = 4;
    int max_features = static_cast<double> (std::sqrt(X_train.cols()));

    RandomForestClassifier model(num_trees, max_depth, min_samples_split, max_features);

    model.fit(X_train, y_train);
    std::cout << "Random Forest model fitted to training data." << "\n";

    Tensor train_predictions = model.predict(X_train);
    Tensor test_predictions = model.predict(X_test);

    double train_accuracy = accuracy_score(y_train, train_predictions);
    double test_accuracy = accuracy_score(y_test, test_predictions);

    double train_f1 = f1_score(y_train, train_predictions);
    double test_f1 = f1_score(y_test, test_predictions);

    std::cout << "Train Accuracy: " << train_accuracy << " || " <<
        "Test Accuracy: " << test_accuracy << "\n";
    
    std::cout << "Train F1: " << train_f1 << " || " << 
        "Test F1: " << test_f1 << "\n";
}
