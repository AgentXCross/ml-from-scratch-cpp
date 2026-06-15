#include "models/logistic_regression.hpp"

#include "core/loss_functions/binary_cross_entropy.hpp"
#include "core/metrics/accuracy.hpp"
#include "preprocessing/dataset.hpp"
#include "preprocessing/standard_scaler.hpp"
#include "preprocessing/train_test_split.hpp"

#include <iostream>

int main(void) {
    Dataset dataset = read_csv_dataset(
        "data/pima-indians-diabetes.csv",
        8,
        false
    );

    DatasetSplit split = train_test_split(dataset, 0.2, true);

    StandardScaler scaler;
    Matrix X_train = scaler.fit_transform(split.train.X);
    Matrix X_test = scaler.transform(split.test.X);

    Matrix y_train = split.train.y;
    Matrix y_test = split.test.y;

    LogisticRegression model(X_train.cols());

    double learning_rate = 0.05;
    int epochs = 10000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        Matrix probabilities = model.predict_probs(X_train);

        double loss = binary_cross_entropy(y_train, probabilities);
        Matrix dL_dpred = binary_cross_entropy_gradient(y_train, probabilities);

        model.backward(X_train, dL_dpred);
        model.step(learning_rate);

        if (epoch % 100 == 0) {
            Matrix train_predictions = model.predict(X_train);
            double train_accuracy = accuracy_score(y_train, train_predictions);

            std::cout << "Epoch: " << epoch << " | Loss: " << loss << " | Train accuracy: " << train_accuracy << "\n";
        }
    }

    Matrix test_probabilities = model.predict_probs(X_test);
    Matrix test_predictions = model.predict(X_test);

    double test_loss = binary_cross_entropy(y_test, test_probabilities);
    double test_accuracy = accuracy_score(y_test, test_predictions);

    std::cout << "\nTest Loss: " << test_loss << "\n";
    std::cout << "\nTest Accuracy: " << test_accuracy << "\n";

    std::cout << "\nFinal Weights\n";
    model.weights().print();

    std::cout << "\nFinal Bias\n";
    model.bias().print();

    return 0;
}