#include "cpp_ml/models/softmax_regression.hpp"

#include "cpp_ml/core/loss_functions/cross_entropy.hpp"
#include "cpp_ml/preprocessing/dataset.hpp"
#include "cpp_ml/preprocessing/standard_scaler.hpp"
#include "cpp_ml/preprocessing/train_test_split.hpp"
#include "cpp_ml/preprocessing/one_hot_encode.hpp"
#include "cpp_ml/core/metrics/accuracy.hpp"

#include <iostream>

int main(void) {
    cpp_ml::Dataset dataset = cpp_ml::read_csv_dataset(
        "data/dry_bean_numeric.csv",
        16,
        true
    );

    cpp_ml::DatasetSplit split = cpp_ml::train_test_split(dataset, 0.2, true);

    cpp_ml::StandardScaler scaler;
    cpp_ml::Tensor X_train = scaler.fit_transform(split.train.X);
    cpp_ml::Tensor X_test = scaler.transform(split.test.X);

    cpp_ml::Tensor y_train = split.train.y;
    cpp_ml::Tensor y_test = split.test.y;

    int num_classes = 7;

    cpp_ml::SoftmaxRegression model(X_train.cols(), num_classes);

    double learning_rate = 0.01;
    int epochs = 5000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        cpp_ml::Tensor probabilities = model.predict_probs(X_train);
        cpp_ml::Tensor y_train_one_hot = one_hot_encode(y_train, num_classes);
        
        double loss = cpp_ml::cross_entropy(y_train_one_hot, probabilities);
        cpp_ml::Tensor dL_dlogits = cpp_ml::cross_entropy_gradient(y_train_one_hot, probabilities);

        model.backward(X_train, dL_dlogits);
        model.step(learning_rate);

        if (epoch % 100 == 0) {
            cpp_ml::Tensor train_predictions = model.predict(X_train);
            double train_accuracy = accuracy_score(y_train, train_predictions);

            std::cout << "Epoch: " << epoch << " | Loss: " << loss << 
                        " | Train accuracy: " << train_accuracy << "\n";
        }
    }

    cpp_ml::Tensor test_probabilities = model.predict_probs(X_test);
    cpp_ml::Tensor test_preds = model.predict(X_test);
    cpp_ml::Tensor y_test_one_hot = cpp_ml::one_hot_encode(y_test, num_classes);

    double test_loss = cpp_ml::cross_entropy(y_test_one_hot, test_probabilities);
    double test_accuracy = cpp_ml::accuracy_score(y_test, test_preds);

    std::cout << "\nTest Loss: " << test_loss << "\n";
    std::cout << "\nTest Accuracy: " << test_accuracy << "\n";

    return 0;
}
