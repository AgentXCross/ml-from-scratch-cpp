from sklearn.datasets import load_digits
import pandas as pd

OUTPUT_PATH = "data/digits.csv"

digits = load_digits()

df = pd.DataFrame(digits.data)
df["label"] = digits.target

df.to_csv(OUTPUT_PATH, index = False)
