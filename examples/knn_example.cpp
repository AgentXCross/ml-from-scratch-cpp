#include "cpp_ml/models/knn.hpp"

#include "cpp_ml/core/metrics/accuracy.hpp"
#include "cpp_ml/preprocessing/dataset.hpp"
#include "cpp_ml/preprocessing/standard_scaler.hpp"
#include "cpp_ml/preprocessing/train_test_split.hpp"

#include <iostream>

int main(void) {
    cpp_ml::Dataset dataset = cpp_ml::read_csv_dataset(
        "data/digits.csv",
        64,
        true
    );

    cpp_ml::DatasetSplit split = cpp_ml::train_test_split(dataset, 0.2, true);

    cpp_ml::StandardScaler scaler;
    cpp_ml::Tensor X_train = scaler.fit_transform(split.train.X);
    cpp_ml::Tensor X_test = scaler.transform(split.test.X);

    cpp_ml::Tensor y_train = split.train.y;
    cpp_ml::Tensor y_test = split.test.y;

    cpp_ml::KNN model(5);

    model.fit(X_train, y_train);

    cpp_ml::Tensor test_preds = model.predict(X_test);

    double accuracy = accuracy_score(y_test, test_preds);

    std::cout << "Test Accuracy: " << accuracy 
            << " on " << X_test.rows() << " samples.\n";

    return 0;
}
