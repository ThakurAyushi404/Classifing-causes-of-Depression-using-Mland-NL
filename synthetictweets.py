import pandas as pd

# ✅ Load the dataset (update the file path)
file_path = "D:/unitec/MachineLearningCourse/Thesis_code/synthetic_tweets.csv" 
df = pd.read_csv(file_path, dtype=str, low_memory=False)

# ✅ Remove duplicate rows based on the "tweet" column
df_cleaned = df.drop_duplicates(subset=['tweet'], keep='first')

# ✅ Save the cleaned dataset
output_path = "cleaned_dataset.csv"
df_cleaned.to_csv(output_path, index=False)

print(f"\n✅ Duplicate rows removed! Cleaned dataset saved as '{output_path}'")
print(f"\n📌 Original dataset size: {len(df)} rows")
print(f"📌 Cleaned dataset size: {len(df_cleaned)} rows")
