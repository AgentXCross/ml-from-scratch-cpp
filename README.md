# C++ Machine Learning Library from Scratch

This project is a small machine learning library built from scratch in C++. I'm implementing classical machine learning algorithms alongside deep learning components. The goal is to build the library in a PyTorch-style, where training loops are explicit and the math behind forward passes, gradient calculations, and parameter updates stays visible.

## Repo Structure
```text
root/
├── build/                      CMake build output.
├── data/                       Small datasets used by examples.
├── examples/                   Model usage examples.
├── include/                    Headers.
├── scripts/                    Python dataset prep and shell scripts.
├── src/                        Source implementations.
│   ├── core/                   Core building blocks.
│   │   ├── activations/        Activation functions and their gradients.
│   │   ├── loss_functions/     Loss functions and their gradients.
│   │   ├── metrics/            Evaluation metrics.
│   │   └── utils/              Helper operations.
│   ├── models/                 Model implementations.
│   ├── optim/                  Optimizers and parameter update logic.
│   └── preprocessing/          Data loading, splitting, and scaling.
└── tests/                      Correctness checks for core components.
```

## Implemented Algorithms
- [x] Linear Regression
- [x] Logistic Regression
- [ ] Ridge / Lasso / Elastic Net (Linear Regression with Regularization)
- [x] Perceptron
- [x] ADALINE
- [x] Softmax / Multinomial Logistic Regression
- [x] KNN
- [x] Naive Bayes
- [x] Decision Trees (Classifier and Regressor)
- [x] Random Forests & Bagging (Classifier and Regressor)
- [ ] Gradient Boosted Trees
- [ ] Linear SVM
- [ ] K-Means Clustering
- [ ] PCA
- [ ] Feedforward Neural Network / MLP
