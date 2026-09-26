#include "core/layers/sequential.hpp"
#include "core/layers/linear.hpp"
#include "core/layers/layer.hpp"

#include "core/activations/relu.hpp"
#include "core/activations/sigmoid.hpp"

#include "core/loss_functions/binary_cross_entropy.hpp"

#include "core/utils/threshold.hpp"

#include "preprocessing/dataset.hpp"
#include "preprocessing/standard_scaler.hpp"
#include "preprocessing/train_test_split.hpp"

#include "core/metrics/accuracy.hpp"
#include "core/metrics/f1_score.hpp"
#include "core/metrics/precision.hpp"
#include "core/metrics/recall.hpp"

#include <iostream>
#include <memory>

int main(void) {
    cpp_ml::Dataset dataset = cpp_ml::read_csv_dataset(
        "data/wisconsin_breast_cancer.csv",
        30,
        true
    );

    cpp_ml::DatasetSplit split = cpp_ml::train_test_split(dataset, 0.2, true);

    cpp_ml::StandardScaler scaler;
    cpp_ml::Tensor X_train = scaler.fit_transform(split.train.X);
    cpp_ml::Tensor X_test = scaler.transform(split.test.X);

    cpp_ml::Tensor y_train = split.train.y;
    cpp_ml::Tensor y_test = split.test.y;

    cpp_ml::Sequential MLP;

    int input_features = X_train.cols();
    int epochs = 1001;
    double learning_rate = 0.1;

    MLP.add(std::make_unique<cpp_ml::Linear>(input_features, 40));
    MLP.add(std::make_unique<cpp_ml::ReLU>());
    MLP.add(std::make_unique<cpp_ml::Linear>(40, 50));
    MLP.add(std::make_unique<cpp_ml::ReLU>());
    MLP.add(std::make_unique<cpp_ml::Linear>(50, 15));
    MLP.add(std::make_unique<cpp_ml::ReLU>());
    MLP.add(std::make_unique<cpp_ml::Linear>(15, 1));
    MLP.add(std::make_unique<cpp_ml::Sigmoid>());

    for (int epoch = 0; epoch < epochs; epoch++) {
        cpp_ml::Tensor y_pred_probs = MLP.forward(X_train);
        cpp_ml::Tensor y_pred_binary = cpp_ml::threshold(y_pred_probs);

        double train_loss = binary_cross_entropy(y_train, y_pred_probs);
        cpp_ml::Tensor dL_dpred = binary_cross_entropy_gradient(y_train, y_pred_probs);

        MLP.backward(dL_dpred);
        MLP.step(learning_rate);

        double train_accuracy = accuracy_score(y_train, y_pred_binary);

        if (epoch % 50 == 0) {
            std::cout << "Epoch: " << epoch << " | " <<  "Training Loss: " 
            << train_loss << " | " << "Train Accuracy: " << train_accuracy << "\n";
        }
    }

    cpp_ml::Tensor y_test_pred_probs = MLP.forward(X_test);
    cpp_ml::Tensor y_test_pred_binary = cpp_ml::threshold(y_test_pred_probs);

    double test_loss = binary_cross_entropy(y_test, y_test_pred_probs);
    double test_accuracy = accuracy_score(y_test, y_test_pred_binary);

    std::cout << "Test Loss: " << test_loss << " | " << "Test Accuracy: " << test_accuracy << std::endl;
}