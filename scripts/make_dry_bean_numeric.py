"""
This scripts converts the label column of the Dry Beans dataset
into a numeric column.
"""

import pandas as pd

INPUT_PATH = "data/dry_bean.csv"
OUTPUT_PATH = "data/dry_bean_numeric.csv"

df = pd.read_csv(INPUT_PATH)

unique_targets = df['Class'].unique().tolist()

mapping = dict();

for index, target in enumerate(unique_targets):
    mapping[target] = index

df['Class'] = df['Class'].map(mapping)

df.to_csv(OUTPUT_PATH, index = False)


