#include "models/perceptron.hpp"

#include "core/metrics/accuracy.hpp"
#include "preprocessing/dataset.hpp"
#include "preprocessing/standard_scaler.hpp"
#include "preprocessing/train_test_split.hpp"

#include <iostream>

int main(void) {
    Dataset dataset = read_csv_dataset(
        "data/setosa_binary.csv",
        4,
        true
    );

    DatasetSplit split = train_test_split(dataset, 0.2, true);

    StandardScaler scaler;
    Matrix X_train = scaler.fit_transform(split.train.X);
    Matrix X_test = scaler.transform(split.test.X);

    Matrix y_train = split.train.y;
    Matrix y_test = split.test.y;

    Perceptron model(X_train.cols());

    double learning_rate = 0.0001;
    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        model.train_epoch(X_train, y_train, learning_rate);

        if (epoch % 100 == 0) {
            Matrix train_predictions = model.predict(X_train);
            double train_accuracy = accuracy_score(y_train, train_predictions);

            std::cout << "Epoch: " << epoch <<
                        " | Train accuracy: " << train_accuracy << "\n";
        }
    }

    Matrix test_predictions = model.predict(X_test);
    double test_accuracy = accuracy_score(y_test, test_predictions);

    std::cout << "\nTest Accuracy: " << test_accuracy << "\n";

    return 0;
}