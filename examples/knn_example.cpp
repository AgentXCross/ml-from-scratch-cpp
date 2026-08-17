#include "models/knn.hpp"

#include "core/metrics/accuracy.hpp"
#include "preprocessing/dataset.hpp"
#include "preprocessing/standard_scaler.hpp"
#include "preprocessing/train_test_split.hpp"

#include <iostream>

int main(void) {
    Dataset dataset = read_csv_dataset(
        "data/digits.csv",
        64,
        true
    );

    DatasetSplit split = train_test_split(dataset, 0.2, true);

    StandardScaler scaler;
    Tensor X_train = scaler.fit_transform(split.train.X);
    Tensor X_test = scaler.transform(split.test.X);

    Tensor y_train = split.train.y;
    Tensor y_test = split.test.y;

    KNN model(5);

    model.fit(X_train, y_train);

    Tensor test_preds = model.predict(X_test);

    double accuracy = accuracy_score(y_test, test_preds);

    std::cout << "Test Accuracy: " << accuracy 
            << " on " << X_test.rows() << " samples.\n";

    return 0;
}
