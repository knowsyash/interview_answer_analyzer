import pandas as pd
import os

os.chdir('d:/interview chat bot/AI_Powered_Interview_Coach_Bot-_for_Job_Preparation/AI_Interview_Bot')

# Check combined_training_data.csv
print("=== Checking combined_training_data.csv ===")
df = pd.read_csv('data/real_dataset_score/combined_training_data.csv', nrows=3)

print("\nColumns:", df.columns.tolist())
print("\nShape:", df.shape)
print("\nFirst row question:", df['question'].iloc[0] if 'question' in df.columns else 'NO QUESTION COL')
print("\nHas 'answer' column:", 'answer' in df.columns)

if 'answer' in df.columns:
    print("\nFirst answer (first 200 chars):")
    ans = df['answer'].iloc[0]
    if pd.notna(ans):
        print(str(ans)[:200])
    else:
        print("NULL/NaN")
    
    print("\nNull count in answer column:", df['answer'].isna().sum())
else:
    print("\nNO ANSWER COLUMN FOUND!")
    print("\nActual columns are:", list(df.columns))

print("\n=== Sample row data ===")
print(df.iloc[0].to_dict())
