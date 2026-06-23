# scripts/run_target_completion.zsh

# source scripts/run_target_completion.zsh to initialize

autoload -Uz compinit
compinit

_run_target_completions() {
    local -a targets

    targets=(
        linear_regression_example
        logistic_regression_example
        perceptron_example
        adaline_example
        knn_example
        softmax_regression_example
        naive_bayes_example

        test_matrix
    )

    compadd -- "${targets[@]}"
}

compdef _run_target_completions ./scripts/run_target.sh
compdef _run_target_completions scripts/run_target.sh
compdef _run_target_completions run_target.sh