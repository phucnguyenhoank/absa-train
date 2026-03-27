import pandas as pd

df = pd.read_csv("multisentiment-uit-vsfc/df_final_train.csv")

df_sampled = df.sample(n=20000)

df_sampled.to_csv(
    "multisentiment-uit-vsfc/df_sampled_final_train.csv", index=False
)
