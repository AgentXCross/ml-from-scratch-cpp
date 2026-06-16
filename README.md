# C++ Machine Learning Library from Scratch

This project is a small machine learning library built from scratch in C++. I'm implementing classical machine learning algorithms alongside deep learning components. The goal is to build the library in a PyTorch-style, where training loops are explicit and the math behind forward passes, gradient calculations, and parameter updates stays visible.

## Source Code Structure
```text
src/
├── core/                   Core building blocks.
│   ├── activations/        Activation functions and their gradients.
│   ├── loss_functions/     Loss functions and their gradients.
│   ├── metrics/            Evaluation metrics.
│   └── utils/              Helper operations.
├── models/                 Trainable model definitions.
├── optim/                  Optimizers (Parameter updates are currently implemented directly within each model)
└── preprocessing/          Data preparation.
```

## Algorithms I Plan to Implement
- [x] Linear Regression
- [x] Logistic Regression
- [ ] Ridge / Lasso / Elastic Net (Linear Regression with Regularization)
- [ ] Perceptron
- [ ] ADALINE
- [ ] KNN
- [ ] Naive Bayes
- [ ] Decision Tree
- [ ] Random Forest
- [ ] Gradient Boosted Trees
- [ ] Linear SVM
- [ ] K-Means Clustering
- [ ] PCA
- [ ] Feedforward Neural Network / MLP