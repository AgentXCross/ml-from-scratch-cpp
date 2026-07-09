import pandas as pd
from sklearn.datasets import load_breast_cancer

OUTPUT_PATH = "data/wisconsin_breast_cancer.csv"

data = load_breast_cancer(as_frame = True)

df = data.frame

df.to_csv(OUTPUT_PATH, index = False)