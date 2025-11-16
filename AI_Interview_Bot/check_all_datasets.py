import pandas as pd
import os

os.chdir('d:/interview chat bot/AI_Powered_Interview_Coach_Bot-_for_Job_Preparation/AI_Interview_Bot')

datasets = [
    'combined_training_data.csv',
    'interview_data_with_scores.csv',
    'stackoverflow_training_data.csv',
    'webdev_interview_qa.csv'
]

print("="*80)
print("CHECKING ALL DATASETS IN data/real_dataset_score/")
print("="*80)

for dataset in datasets:
    path = f'data/real_dataset_score/{dataset}'
    if os.path.exists(path):
        print(f"\n{'='*80}")
        print(f"📁 {dataset}")
        print(f"{'='*80}")
        
        df = pd.read_csv(path, nrows=2)
        
        print(f"\n✅ Columns: {df.columns.tolist()}")
        print(f"📊 Total rows: {pd.read_csv(path).shape[0]}")
        
        # Check for answer-related columns
        answer_cols = [col for col in df.columns if 'answer' in col.lower()]
        print(f"\n🔍 Answer-related columns: {answer_cols if answer_cols else 'NONE FOUND ❌'}")
        
        # Check for question column
        question_cols = [col for col in df.columns if 'question' in col.lower()]
        print(f"❓ Question columns: {question_cols}")
        
        # Show first row sample
        print(f"\n📝 Sample data (first row):")
        for col in df.columns[:6]:  # Show first 6 columns
            val = df[col].iloc[0]
            if isinstance(val, str) and len(val) > 100:
                val = val[:100] + "..."
            print(f"   {col}: {val}")
        
        # Check if answer column has data
        if answer_cols:
            for ans_col in answer_cols:
                null_count = pd.read_csv(path)[ans_col].isna().sum()
                total = pd.read_csv(path).shape[0]
                print(f"\n   ✓ {ans_col}: {total - null_count}/{total} non-null ({((total-null_count)/total*100):.1f}%)")
    else:
        print(f"\n❌ {dataset} - FILE NOT FOUND!")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
