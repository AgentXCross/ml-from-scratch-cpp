# C++ Machine Learning Library from Scratch

This project is a small machine learning library built from scratch in C++. I'm implementing classical machine learning algorithms such as linear regression and logistic regression, alongside deep learning components. The goal is to build the library in a more PyTorch-style way, where training loops are explicit and the mathematics behind forward passes, gradient calculations, and parameter updates stays visible.

## Source Structure
```text
src/
├── core/                   Core math and ML building blocks.
│   ├── activations/        Activation functions.
│   ├── loss_functions/     Loss functions and their gradients.
│   ├── metrics/            Evaluation metrics.
│   └── utils/              Helper operations.
├── models/                 Trainable models.
├── optim/             Optimizers and parameter update.
└── preprocessing/     Data preparation.
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