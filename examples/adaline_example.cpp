#include "models/adaline.hpp"

#include "core/loss_functions/mse.hpp"
#include "core/metrics/accuracy.hpp"
#include "preprocessing/dataset.hpp"
#include "preprocessing/standard_scaler.hpp"
#include "preprocessing/train_test_split.hpp"

#include <iostream>

int main(void) {
    cpp_ml::Dataset dataset = cpp_ml::read_csv_dataset(
        "data/setosa_pm1.csv",
        4,
        true
    );

    cpp_ml::DatasetSplit split = train_test_split(dataset, 0.2, true);

    cpp_ml::StandardScaler scaler;
    cpp_ml::Tensor X_train = scaler.fit_transform(split.train.X);
    cpp_ml::Tensor X_test = scaler.transform(split.test.X);

    cpp_ml::Tensor y_train = split.train.y;
    cpp_ml::Tensor y_test = split.test.y;

    cpp_ml::ADALINE model(X_train.cols());

    double learning_rate = 0.005;
    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        cpp_ml::Tensor raw = model.predict_raw(X_train);

        double loss = mean_squared_error(y_train, raw);
        cpp_ml::Tensor dL_dpred = mean_squared_error_gradient(y_train, raw);

        model.backward(X_train, dL_dpred);
        model.step(learning_rate);

        if (epoch % 100 == 0) {
            cpp_ml::Tensor train_preds = model.predict(X_train);
            double train_accuracy = accuracy_score(y_train, train_preds);

            std::cout << "Epoch: " << epoch << " | Loss: " << loss << 
                        " | Train accuracy: " << train_accuracy << "\n";
        }
    }

    cpp_ml::Tensor test_raw = model.predict_raw(X_test);
    cpp_ml::Tensor test_predictions = model.predict(X_test);

    double test_loss = mean_squared_error(y_test, test_raw);
    double test_accuracy = accuracy_score(y_test, test_predictions);

    std::cout << "\nTest Loss: " << test_loss << "\n";
    std::cout << "\nTest Accuracy: " << test_accuracy << "\n";

    std::cout << "\nFinal Weights\n";
    model.weights().print();

    std::cout << "\nFinal Bias\n";
    model.bias().print();

    return 0;
}