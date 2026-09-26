#include "cpp_ml/models/perceptron.hpp"

#include "cpp_ml/core/metrics/accuracy.hpp"
#include "cpp_ml/preprocessing/dataset.hpp"
#include "cpp_ml/preprocessing/standard_scaler.hpp"
#include "cpp_ml/preprocessing/train_test_split.hpp"

#include <iostream>

int main(void) {
    cpp_ml::Dataset dataset = cpp_ml::read_csv_dataset(
        "data/setosa_binary.csv",
        4,
        true
    );

    cpp_ml::DatasetSplit split = cpp_ml::train_test_split(dataset, 0.2, true);

    cpp_ml::StandardScaler scaler;
    cpp_ml::Tensor X_train = scaler.fit_transform(split.train.X);
    cpp_ml::Tensor X_test = scaler.transform(split.test.X);

    cpp_ml::Tensor y_train = split.train.y;
    cpp_ml::Tensor y_test = split.test.y;

    cpp_ml::Perceptron model(X_train.cols());

    double learning_rate = 0.0001;
    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        model.train_epoch(X_train, y_train, learning_rate);

        if (epoch % 100 == 0) {
            cpp_ml::Tensor train_predictions = model.predict(X_train);
            double train_accuracy = accuracy_score(y_train, train_predictions);

            std::cout << "Epoch: " << epoch <<
                        " | Train accuracy: " << train_accuracy << "\n";
        }
    }

    cpp_ml::Tensor test_predictions = model.predict(X_test);
    double test_accuracy = accuracy_score(y_test, test_predictions);

    std::cout << "\nTest Accuracy: " << test_accuracy << "\n";

    return 0;
}
