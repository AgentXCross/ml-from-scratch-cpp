"""
This script takes the Iris dataset from iris.csv and transforms it into a 
binary dataset where the setosa class = 1 and all non-setosa classes = 0.
"""

import pandas as pd

INPUT_PATH = "data/iris.csv"
OUTPUT_PATH = "data/setosa_binary.csv"

df = pd.read_csv(INPUT_PATH)

# Species column is the last one
species_column = df.columns[-1]

df[species_column] = df[species_column].apply(
    lambda value: 1 if value == "setosa" else 0
)

df.to_csv(OUTPUT_PATH, index = False)
