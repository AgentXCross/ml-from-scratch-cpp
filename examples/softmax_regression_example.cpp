#include "models/softmax_regression.hpp"

#include "core/loss_functions/cross_entropy.hpp"
#include "preprocessing/dataset.hpp"
#include "preprocessing/standard_scaler.hpp"
#include "preprocessing/train_test_split.hpp"
#include "preprocessing/one_hot_encode.hpp"
#include "core/metrics/accuracy.hpp"

#include <iostream>

int main(void) {
    Dataset dataset = read_csv_dataset(
        "data/dry_bean_numeric.csv",
        16,
        true
    );

    DatasetSplit split = train_test_split(dataset, 0.2, true);

    StandardScaler scaler;
    Matrix X_train = scaler.fit_transform(split.train.X);
    Matrix X_test = scaler.transform(split.test.X);

    Matrix y_train = split.train.y;
    Matrix y_test = split.test.y;

    int num_classes = 7;

    SoftmaxRegression model(X_train.cols(), num_classes);

    double learning_rate = 0.01;
    int epochs = 5000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        Matrix probabilities = model.predict_probs(X_train);
        Matrix y_train_one_hot = one_hot_encode(y_train, num_classes);
        
        double loss = cross_entropy(y_train_one_hot, probabilities);
        Matrix dL_dlogits = cross_entropy_gradient(y_train_one_hot, probabilities);

        model.backward(X_train, dL_dlogits);
        model.step(learning_rate);

        if (epoch % 100 == 0) {
            Matrix train_predictions = model.predict(X_train);
            double train_accuracy = accuracy_score(y_train, train_predictions);

            std::cout << "Epoch: " << epoch << " | Loss: " << loss << 
                        " | Train accuracy: " << train_accuracy << "\n";
        }
    }

    Matrix test_probabilities = model.predict_probs(X_test);
    Matrix test_preds = model.predict(X_test);
    Matrix y_test_one_hot = one_hot_encode(y_test, num_classes);

    double test_loss = cross_entropy(y_test_one_hot, test_probabilities);
    double test_accuracy = accuracy_score(y_test, test_preds);

    std::cout << "\nTest Loss: " << test_loss << "\n";
    std::cout << "\nTest Accuracy: " << test_accuracy << "\n";

    return 0;
}