#pragma once

// Core
#include "cpp_ml/core/tensor.hpp"
#include "cpp_ml/core/matrix.hpp"

// Activations
#include "cpp_ml/core/activations/relu.hpp"
#include "cpp_ml/core/activations/sigmoid.hpp"
#include "cpp_ml/core/activations/tanh.hpp"
#include "cpp_ml/core/activations/leaky_relu.hpp"
#include "cpp_ml/core/activations/gelu.hpp"
#include "cpp_ml/core/activations/softmax.hpp"

// Layers
#include "cpp_ml/core/layers/layer.hpp"
#include "cpp_ml/core/layers/linear.hpp"
#include "cpp_ml/core/layers/sequential.hpp"

// Loss Functions
#include "cpp_ml/core/loss_functions/mse.hpp"
#include "cpp_ml/core/loss_functions/mae.hpp"
#include "cpp_ml/core/loss_functions/binary_cross_entropy.hpp"
#include "cpp_ml/core/loss_functions/cross_entropy.hpp"
#include "cpp_ml/core/loss_functions/huber_loss.hpp"

// Metrics
#include "cpp_ml/core/metrics/accuracy.hpp"
#include "cpp_ml/core/metrics/precision.hpp"
#include "cpp_ml/core/metrics/recall.hpp"
#include "cpp_ml/core/metrics/f1_score.hpp"
#include "cpp_ml/core/metrics/r2_score.hpp"
#include "cpp_ml/core/metrics/confusion_matrix.hpp"

// Utils
#include "cpp_ml/core/utils/threshold.hpp"
#include "cpp_ml/core/utils/euclidean_distance.hpp"
#include "cpp_ml/core/utils/tree_utils.hpp"

// Preprocessing
#include "cpp_ml/preprocessing/dataset.hpp"
#include "cpp_ml/preprocessing/train_test_split.hpp"
#include "cpp_ml/preprocessing/standard_scaler.hpp"
#include "cpp_ml/preprocessing/min_max_scaler.hpp"
#include "cpp_ml/preprocessing/one_hot_encode.hpp"

// Models
#include "cpp_ml/models/linear_regression.hpp"
#include "cpp_ml/models/logistic_regression.hpp"
#include "cpp_ml/models/softmax_regression.hpp"
#include "cpp_ml/models/perceptron.hpp"
#include "cpp_ml/models/adaline.hpp"
#include "cpp_ml/models/knn.hpp"
#include "cpp_ml/models/knn_regressor.hpp"
#include "cpp_ml/models/gaussian_naive_bayes.hpp"
#include "cpp_ml/models/decision_tree_classifier.hpp"
#include "cpp_ml/models/decision_tree_regressor.hpp"
#include "cpp_ml/models/random_forest_classifier.hpp"