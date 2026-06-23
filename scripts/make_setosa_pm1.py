"""
This script takes the Iris dataset from iris.csv and transforms it into a 
binary dataset where the setosa class = 1 and all non-setosa classes = -1.

+ or - 1 is the standard for the ADALINE classifier.
"""

import pandas as pd

INPUT_PATH = "data/iris.csv"
OUTPUT_PATH = "data/setosa_pm1.csv"

df = pd.read_csv(INPUT_PATH)

# Species/label is the last column
species_column = df.columns[-1]

df[species_column] = df[species_column].apply(
    lambda value: 1 if value in ["setosa", "Iris-setosa"] else -1
)

df.to_csv(OUTPUT_PATH, index = False)