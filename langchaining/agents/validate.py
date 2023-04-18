import pandas as pd

from langchaining.helpers import BASE_DIR

df = pd.read_csv(BASE_DIR / "agents/action.csv")
row_num = df["gross(in $)"].idxmax()
print(df.iloc[row_num])
print(df.iloc[0])
