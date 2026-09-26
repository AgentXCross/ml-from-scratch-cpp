#include "cpp_ml/models/logistic_regression.hpp"

#include "cpp_ml/core/loss_functions/binary_cross_entropy.hpp"
#include "cpp_ml/core/metrics/accuracy.hpp"
#include "cpp_ml/preprocessing/dataset.hpp"
#include "cpp_ml/preprocessing/standard_scaler.hpp"
#include "cpp_ml/preprocessing/train_test_split.hpp"

#include <iostream>

int main(void) {
    cpp_ml::Dataset dataset = cpp_ml::read_csv_dataset(
        "data/pima-indians-diabetes.csv",
        8,
        false
    );

    cpp_ml::DatasetSplit split = train_test_split(dataset, 0.2, true);

    cpp_ml::StandardScaler scaler;
    cpp_ml::Tensor X_train = scaler.fit_transform(split.train.X);
    cpp_ml::Tensor X_test = scaler.transform(split.test.X);

    cpp_ml::Tensor y_train = split.train.y;
    cpp_ml::Tensor y_test = split.test.y;

    cpp_ml::LogisticRegression model(X_train.cols());

    double learning_rate = 0.05;
    int epochs = 10000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        cpp_ml::Tensor probabilities = model.predict_probs(X_train);

        double loss = binary_cross_entropy(y_train, probabilities);
        cpp_ml::Tensor dL_dpred = binary_cross_entropy_gradient(y_train, probabilities);

        model.backward(X_train, dL_dpred);
        model.step(learning_rate);

        if (epoch % 100 == 0) {
            cpp_ml::Tensor train_predictions = model.predict(X_train);
            double train_accuracy = accuracy_score(y_train, train_predictions);

            std::cout << "Epoch: " << epoch << " | Loss: " << loss << 
                        " | Train accuracy: " << train_accuracy << "\n";
        }
    }

    cpp_ml::Tensor test_probabilities = model.predict_probs(X_test);
    cpp_ml::Tensor test_predictions = model.predict(X_test);

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
