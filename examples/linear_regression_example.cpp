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
        {10.0},
        {11.0},
        {12.0},
        {13.0},
        {14.0},
        {15.0},
        {20.0},
        {25.0},
        {50.0},
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
        {21.0},
        {23.0},
        {25.0},
        {27.0},
        {29.0},
        {31.0},
        {41.0},
        {51.0},
        {101.0}
    });

    LinearRegression model(1);

    double learning_rate = 0.001;
    int epochs = 10000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        Matrix predictions = model.predict(X);

        double loss = mean_squared_error(y, predictions);
        Matrix loss_gradient = mean_squared_error_gradient(y, predictions);

        model.backward(X, loss_gradient);
        model.step(learning_rate);

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch
                        << " | Loss: " << loss
                        << "\n";
        }
    }

    std::cout << "\nFinal weights:\n";
    model.weights().print();

    std::cout << "\nFinal bias:\n";
    model.bias().print();

    std::cout << "\nPredications:\n";
    model.predict(X).print();

    return 0;
}