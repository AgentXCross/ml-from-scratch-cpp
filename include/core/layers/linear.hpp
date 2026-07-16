#pragma once

#include "core/layers/layer.hpp"

class Linear : public Layer {
private:
    Matrix weights_; // W: (in_features, out_features)
    Matrix bias_; // b: (1, out_features)

    Matrix input_; // X: (n_samples, in_features)

    Matrix dL_dW_; // dL_dW: (in_features, out_features)
    Matrix dL_db_; // dL/db: (1, out_features)

public:
    Linear();
    Linear(
        int in_features,
        int out_features
    );

    /*
    Forward pass through a linear layer.
    out = XW + b for all samples in X 
    For a single sample row-vector x, out = xW + b

    Shapes:
        X: (n_samples, in_features)
        x: (1, in_features)
        W: (in_features, out_features)
        b: (1, out_features)
        out: (n_samples, out_features)
    */
    Matrix forward(const Matrix &X) override;

    /*
    Backpropagation through a linear layer.

    dL_dout is the gradient of the loss with respect to the output
    and has the same shape as the output of the layer.

    Returns dL_dX is the gradient of the loss with respect to the input
    and is used for previous layers. 

    dL_dW = X.T @ dL_dout
    dL_db = sum of dL_dout over all samples
    dL_dX = dL_dout @ W.T

    Ex: Let x = [x1 x2], W = [w11 w12] and b = [b1 b2]
                             [w21 w22]
    
        out = xW + b = [x1w11 + x2w21 + b1  x1w12 + x2w22 + b2]
        dL_dout = [dL_dout1 dL_dout2]

        dL_dw11 = dL_dout1 * dout1_dw11 = dL_dout1 * x1 and repeat for all other weights

        dL_db1 = dL_dout1 * dout1_db1 = dL_dout1

        Since changing the inputs affects both outputs,
        dL_dx1 = dL_dout1 * dout1_dx1 + dL_dout2 * dout2_dx1
               = dL_dout1 * w11 + dL_dout2 * w12
    */
    Matrix backward(const Matrix &dL_dout) override;

    void step(double learning_rate) override;

    Matrix weights() const;
    Matrix bias() const;
};