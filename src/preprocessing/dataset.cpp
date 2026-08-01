#include "preprocessing/dataset.hpp"

#include <cassert>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>


Dataset read_csv_dataset(
    const std::string &filepath,
    int label_column,
    bool has_header
) {
    std::ifstream file(filepath);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + filepath);
    }

    std::vector<std::vector<double>> X_values;
    std::vector<std::vector<double>> y_values;

    std::string line;

    if (has_header) {
        std::getline(file, line); // get the line to skip it
    }

    while (std::getline(file, line)) { // loop through all samples
        if (line.empty()) {
            continue;
        }

        std::stringstream line_stream(line);
        std::string cell;
        std::vector<double> row_values;

        while (std::getline(line_stream, cell, ',')) {
            row_values.push_back(std::stod(cell)); // convert all to double
        }

        if (row_values.empty()) {
            continue;
        }

        if (label_column < 0 || label_column >= static_cast<int> (row_values.size())) {
            throw std::invalid_argument("label_column is out of range for CSV row");
        }

        std::vector<double> X_row;
        std::vector<double> y_row;

        for (int j = 0; j < static_cast<int> (row_values.size()); j++) {
            if (j == label_column) {
                y_row.push_back(row_values[j]);
            } else {
                X_row.push_back(row_values[j]);
            }
        }

        X_values.push_back(X_row);
        y_values.push_back(y_row);
    }

    if (X_values.empty() || y_values.empty()) {
        throw std::invalid_argument("CSV file did not contain any data rows");
    }

    Dataset dataset;
    dataset.X = Tensor::from_vector(X_values);
    dataset.y = Tensor::from_vector(y_values);

    assert(dataset.X.is_matrix());
    assert(dataset.y.is_matrix());
    assert(dataset.X.cols() > 0);
    assert(dataset.X.rows() == dataset.y.rows());
    assert(dataset.y.cols() == 1);

    return dataset;
}