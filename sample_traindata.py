import pandas as pd

# Load the dataset
df = pd.read_csv("multisentiment-uit-vsfc/df_final_train.csv")

# Sample 40,000 rows
# random_state ensures your results are reproducible
df_sampled = df.sample(n=30000)

# Optional: Save the sample to a new file
df_sampled.to_csv(
    "multisentiment-uit-vsfc/df_sampled_final_train.csv", index=False
)
