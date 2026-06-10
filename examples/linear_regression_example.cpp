#include "models/linear_regression.hpp"
#include "core/matrix.hpp"
#include "core/loss_functions/mse.hpp"

#include <iostream>

// Test for LinearRegression using a y = 2x + 1 dataset

int main() {
    // Dataset: y = 2x + 1
    Matrix X = Matrix::from_vector({
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0},
        {6.0},
        {7.0},
        {8.0},
        {9.0},
        {10.0}
    });

    Matrix y = Matrix::from_vector({
        {3.0},
        {5.0},
        {7.0},
        {9.0},
        {11.0},
        {13.0},
        {15.0},
        {17.0},
        {19.0},
        {21.0}
    });

    LinearRegression model(1);

    double learning_rate = 0.001;
    int epochs = 500;

    for (int epoch = 0; epoch < epochs; epoch++) {
        Matrix predictions = model.predict(X);
    }
}