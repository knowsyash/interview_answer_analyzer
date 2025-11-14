"""
Comprehensive Dataset Analysis for AI Interview Bot
Analyzes all real production datasets used by the application
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import os

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("AI INTERVIEW BOT - DATASET ANALYSIS")
print("="*70)

# ============================================================================
# SECTION 1: LOAD ALL DATASETS
# ============================================================================

print("\n" + "="*70)
print("SECTION 1: LOADING DATASETS")
print("="*70)

# Dataset paths
datasets = {
    'behavioral': 'real_dataset_score/interview_data_with_scores.csv',
    'webdev': 'real_dataset_score/webdev_interview_qa.csv',
    'stackoverflow': 'real_dataset_score/stackoverflow_training_data.csv',
    'combined': 'real_dataset_score/combined_training_data.csv'
}

data = {}
for name, path in datasets.items():
    if os.path.exists(path):
        data[name] = pd.read_csv(path)
        print(f"OK {name.upper():15} : {len(data[name]):6} records")
    else:
        print(f"ERROR {name.upper():15} : NOT FOUND")

# ============================================================================
# SECTION 2: BEHAVIORAL DATASET ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("SECTION 2: BEHAVIORAL DATASET ANALYSIS")
print("="*70)

behavioral = data['behavioral']

print(f"\n📊 Dataset Overview:")
print(f"   Records: {len(behavioral)}")
print(f"   Columns: {list(behavioral.columns)}")

print(f"\n📈 Score Distribution:")
print(behavioral['human_score'].value_counts().sort_index())

print(f"\n📊 Score Statistics:")
print(f"   Mean: {behavioral['human_score'].mean():.2f}")
print(f"   Median: {behavioral['human_score'].median():.2f}")
print(f"   Std Dev: {behavioral['human_score'].std():.2f}")
print(f"   Min: {behavioral['human_score'].min()}")
print(f"   Max: {behavioral['human_score'].max()}")

# Analyze competencies
print(f"\n🎯 Competency Analysis:")
all_competencies = []
for comp in behavioral['competency']:
    # Parse competency string (stored as string representation of list)
    if isinstance(comp, str) and '[' in comp:
        comps = [c.strip().strip("'\"[]") for c in comp.split(',')]
        all_competencies.extend(comps)

competency_counts = Counter(all_competencies)
print(f"\n   Total unique competencies: {len(competency_counts)}")
print(f"\n   Top 10 Competencies:")
for comp, count in competency_counts.most_common(10):
    print(f"      {comp:30} : {count:4} occurrences")

# Answer length analysis
behavioral['answer_length'] = behavioral['answer'].str.split().str.len()
print(f"\n📝 Answer Length Statistics:")
print(f"   Mean words: {behavioral['answer_length'].mean():.1f}")
print(f"   Median words: {behavioral['answer_length'].median():.1f}")
print(f"   Min words: {behavioral['answer_length'].min()}")
print(f"   Max words: {behavioral['answer_length'].max()}")

# ============================================================================
# SECTION 3: WEB DEVELOPMENT DATASET ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("SECTION 3: WEB DEVELOPMENT DATASET ANALYSIS")
print("="*70)

webdev = data['webdev']

print(f"\n📊 Dataset Overview:")
print(f"   Records: {len(webdev)}")
print(f"   Columns: {list(webdev.columns)}")

print(f"\n📈 Score Distribution:")
print(webdev['human_score'].value_counts().sort_index())

print(f"\n📊 Score Statistics:")
print(f"   Mean: {webdev['human_score'].mean():.2f}")
print(f"   Median: {webdev['human_score'].median():.2f}")
print(f"   Std Dev: {webdev['human_score'].std():.2f}")
print(f"   Min: {webdev['human_score'].min()}")
print(f"   Max: {webdev['human_score'].max()}")

# Analyze topics
print(f"\n🎯 Topic Analysis:")
all_topics = []
for comp in webdev['competency']:
    if isinstance(comp, str) and '[' in comp:
        topics = [c.strip().strip("'\"[]") for c in comp.split(',')]
        all_topics.extend(topics)

topic_counts = Counter(all_topics)
print(f"\n   Top Topics:")
for topic, count in topic_counts.most_common(15):
    print(f"      {topic:30} : {count:4} occurrences")

# Answer length analysis
webdev['answer_length'] = webdev['answer'].str.split().str.len()
print(f"\n📝 Answer Length Statistics:")
print(f"   Mean words: {webdev['answer_length'].mean():.1f}")
print(f"   Median words: {webdev['answer_length'].median():.1f}")
print(f"   Min words: {webdev['answer_length'].min()}")
print(f"   Max words: {webdev['answer_length'].max()}")

# ============================================================================
# SECTION 4: STACK OVERFLOW DATASET ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("SECTION 4: STACK OVERFLOW DATASET ANALYSIS")
print("="*70)

stackoverflow = data['stackoverflow']

print(f"\n📊 Dataset Overview:")
print(f"   Records: {len(stackoverflow)}")
print(f"   Columns: {list(stackoverflow.columns)}")

print(f"\n📈 Normalized Score Distribution:")
score_bins = pd.cut(stackoverflow['score'], bins=[0, 1, 2, 3, 4, 5], include_lowest=True)
print(score_bins.value_counts().sort_index())

print(f"\n📊 Score Statistics:")
print(f"   Mean: {stackoverflow['score'].mean():.2f}")
print(f"   Median: {stackoverflow['score'].median():.2f}")
print(f"   Std Dev: {stackoverflow['score'].std():.2f}")
print(f"   Min: {stackoverflow['score'].min():.2f}")
print(f"   Max: {stackoverflow['score'].max():.2f}")

print(f"\n📊 Original Stack Overflow Scores:")
print(f"   Mean: {stackoverflow['original_score'].mean():.1f}")
print(f"   Median: {stackoverflow['original_score'].median():.1f}")
print(f"   Min: {stackoverflow['original_score'].min()}")
print(f"   Max: {stackoverflow['original_score'].max()}")

# Analyze tags (domains)
print(f"\n🏷️  Tag/Domain Analysis:")
all_tags = []
for tags_str in stackoverflow['tags']:
    if pd.notna(tags_str):
        tags = [t.strip() for t in str(tags_str).split(',')]
        all_tags.extend(tags)

tag_counts = Counter(all_tags)
print(f"\n   Total unique tags: {len(tag_counts)}")
print(f"\n   Top 30 Tags:")
for tag, count in tag_counts.most_common(30):
    print(f"      {tag:30} : {count:4} questions")

# Domain categorization
domains = {
    'Python': ['python', 'django', 'flask', 'pandas', 'numpy', 'scipy'],
    'Java': ['java', 'spring', 'hibernate', 'maven', 'android'],
    'C#/.NET': ['c#', '.net', 'asp.net', 'entity-framework', 'wpf', 'visual-studio'],
    'JavaScript': ['javascript', 'node.js', 'react', 'angular', 'vue.js', 'typescript'],
    'Database': ['sql', 'mysql', 'postgresql', 'mongodb', 'database', 'oracle'],
    'C++': ['c++', 'stl', 'templates'],
    'Web': ['html', 'css', 'jquery', 'ajax', 'web'],
}

print(f"\n📊 Questions by Domain:")
for domain_name, keywords in domains.items():
    count = sum(1 for tags_str in stackoverflow['tags'] if pd.notna(tags_str) and 
                any(kw in str(tags_str).lower() for kw in keywords))
    print(f"   {domain_name:15} : {count:5} questions")

# Answer length analysis
stackoverflow['answer_length'] = stackoverflow['user_answer'].str.split().str.len()
print(f"\n📝 Answer Length Statistics:")
print(f"   Mean words: {stackoverflow['answer_length'].mean():.1f}")
print(f"   Median words: {stackoverflow['answer_length'].median():.1f}")
print(f"   Min words: {stackoverflow['answer_length'].min()}")
print(f"   Max words: {stackoverflow['answer_length'].max()}")

# ============================================================================
# SECTION 5: COMBINED DATASET ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("SECTION 5: COMBINED TRAINING DATASET ANALYSIS")
print("="*70)

combined = data['combined']

print(f"\n📊 Dataset Overview:")
print(f"   Records: {len(combined)}")
print(f"   Columns: {list(combined.columns)}")
print(f"   Composition: {len(stackoverflow)} SO + {len(behavioral)} Behavioral = {len(combined)}")

print(f"\n📈 Score Distribution:")
score_bins = pd.cut(combined['score'], bins=[0, 1, 2, 3, 4, 5], include_lowest=True)
print(score_bins.value_counts().sort_index())

print(f"\n📊 Score Statistics:")
print(f"   Mean: {combined['score'].mean():.2f}")
print(f"   Median: {combined['score'].median():.2f}")
print(f"   Std Dev: {combined['score'].std():.2f}")
print(f"   Min: {combined['score'].min():.2f}")
print(f"   Max: {combined['score'].max():.2f}")

# Answer length analysis
combined['answer_length'] = combined['user_answer'].str.split().str.len()
print(f"\n📝 Answer Length Statistics:")
print(f"   Mean words: {combined['answer_length'].mean():.1f}")
print(f"   Median words: {combined['answer_length'].median():.1f}")
print(f"   Min words: {combined['answer_length'].min()}")
print(f"   Max words: {combined['answer_length'].max()}")

# ============================================================================
# SECTION 6: CROSS-DATASET COMPARISON
# ============================================================================

print("\n" + "="*70)
print("SECTION 6: CROSS-DATASET COMPARISON")
print("="*70)

comparison = pd.DataFrame({
    'Dataset': ['Behavioral', 'Web Dev', 'Stack Overflow', 'Combined'],
    'Records': [len(behavioral), len(webdev), len(stackoverflow), len(combined)],
    'Avg Score': [
        behavioral['human_score'].mean(),
        webdev['human_score'].mean(),
        stackoverflow['score'].mean(),
        combined['score'].mean()
    ],
    'Avg Words': [
        behavioral['answer_length'].mean(),
        webdev['answer_length'].mean(),
        stackoverflow['answer_length'].mean(),
        combined['answer_length'].mean()
    ]
})

print("\n" + comparison.to_string(index=False))

# ============================================================================
# SECTION 7: DATA QUALITY CHECKS
# ============================================================================

print("\n" + "="*70)
print("SECTION 7: DATA QUALITY CHECKS")
print("="*70)

def check_quality(df, name):
    print(f"\n{name}:")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Duplicate rows: {df.duplicated().sum()}")
    if 'question' in df.columns:
        print(f"   Empty questions: {(df['question'].str.len() == 0).sum()}")
    if 'answer' in df.columns:
        print(f"   Empty answers: {(df['answer'].str.len() == 0).sum()}")
    elif 'user_answer' in df.columns:
        print(f"   Empty answers: {(df['user_answer'].str.len() == 0).sum()}")

check_quality(behavioral, "Behavioral Dataset")
check_quality(webdev, "Web Dev Dataset")
check_quality(stackoverflow, "Stack Overflow Dataset")
check_quality(combined, "Combined Dataset")

# ============================================================================
# SECTION 8: SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("SECTION 8: OVERALL SUMMARY")
print("="*70)

total_qa = len(behavioral) + len(webdev) + len(stackoverflow)
print(f"\n📊 Total Q&A Pairs Available: {total_qa:,}")
print(f"   - Behavioral (STAR): {len(behavioral):,} ({len(behavioral)/total_qa*100:.1f}%)")
print(f"   - Web Development: {len(webdev):,} ({len(webdev)/total_qa*100:.1f}%)")
print(f"   - Stack Overflow: {len(stackoverflow):,} ({len(stackoverflow)/total_qa*100:.1f}%)")

print(f"\n🎯 Training Dataset: {len(combined):,} samples")
print(f"   (Used for Random Forest model training)")

print(f"\n🏷️  Coverage:")
print(f"   - Behavioral Competencies: {len(competency_counts)} unique")
print(f"   - Web Development Topics: {len(topic_counts)} unique")
print(f"   - Technical Tags: {len(tag_counts)} unique")

print(f"\n✅ All datasets loaded and analyzed successfully!")
print("="*70)

# ============================================================================
# SECTION 9: SAVE ANALYSIS RESULTS
# ============================================================================

print("\n" + "="*70)
print("SECTION 9: SAVING ANALYSIS RESULTS")
print("="*70)

# Create output directory
os.makedirs('outputs', exist_ok=True)

# Save summary statistics
summary = {
    'total_qa_pairs': int(total_qa),
    'datasets': {
        'behavioral': {
            'records': int(len(behavioral)),
            'avg_score': float(behavioral['human_score'].mean()),
            'avg_words': float(behavioral['answer_length'].mean()),
            'unique_competencies': len(competency_counts)
        },
        'webdev': {
            'records': int(len(webdev)),
            'avg_score': float(webdev['human_score'].mean()),
            'avg_words': float(webdev['answer_length'].mean()),
            'unique_topics': len(topic_counts)
        },
        'stackoverflow': {
            'records': int(len(stackoverflow)),
            'avg_score': float(stackoverflow['score'].mean()),
            'avg_words': float(stackoverflow['answer_length'].mean()),
            'unique_tags': len(tag_counts)
        },
        'combined': {
            'records': int(len(combined)),
            'avg_score': float(combined['score'].mean()),
            'avg_words': float(combined['answer_length'].mean())
        }
    }
}

with open('outputs/dataset_analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✅ Analysis summary saved to: outputs/dataset_analysis_summary.json")
print("="*70)
