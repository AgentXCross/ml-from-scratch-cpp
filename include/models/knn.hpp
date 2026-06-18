#pragma once

#include "core/matrix.hpp"

class KNN {
private:
    Matrix X_train_;
    Matrix y_train_;
    int k_; // # of neighbors
    bool fitted_;

public:
    KNN();
    KNN(int k);

    void fit(
        const Matrix &X_train,
        const Matrix &y_train
    );

    Matrix predict(const Matrix &X) const;

    int k() const;
};