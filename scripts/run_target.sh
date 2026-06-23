#!/usr/bin/env bash

# chmod +x scripts/run_target.sh to make this file executable

set -e

target="$1"

valid_targets=(
    linear_regression_example
    logistic_regression_example
    perceptron_example
    adaline_example
    knn_example
    softmax_regression_example
    naive_bayes_example

    test_matrix
)

# if string length is 0
if [ -z "$target" ]; then 
    echo "Must provide target in this format: scripts/run_target.sh <target>"
    exit 1
fi

found=false

for t in "${valid_targets[@]}"; do
    if [ "$t" = "$target" ]; then
        found=true
        break
    fi
done

if [ "$found" = "false" ]; then
    echo "Unknown target: $target"
    exit 1;
fi

cmake --build build --target "$target"
./build/"$target"