"""
Process Stack Overflow Archive Data for Random Forest Training

This script extracts question-answer pairs from the archive folder and
prepares them for training the Random Forest scoring model.
"""

import pandas as pd
import os
import re
from html import unescape
from bs4 import BeautifulSoup

# Paths
ARCHIVE_PATH = r"AI_Interview_Bot\archive"
OUTPUT_PATH = r"AI_Interview_Bot\data"
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "stackoverflow_training_data.csv")

def clean_html(text):
    """Remove HTML tags and clean text"""
    if pd.isna(text):
        return ""
    # Parse HTML
    soup = BeautifulSoup(str(text), 'html.parser')
    # Extract text
    text = soup.get_text()
    # Unescape HTML entities
    text = unescape(text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_score(score, min_score=-5, max_score=100):
    """Normalize answer score to 0-10 range"""
    # Clip to reasonable range
    score = max(min_score, min(score, max_score))
    # Normalize to 0-10
    normalized = ((score - min_score) / (max_score - min_score)) * 10
    return round(normalized, 1)

def filter_interview_related(tags_list):
    """Check if tags are interview/technical skill related"""
    interview_keywords = {
        'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'html', 'css',
        'react', 'node.js', 'django', 'flask', 'angular', 'vue.js',
        'algorithm', 'data-structures', 'database', 'web-development',
        'machine-learning', 'api', 'rest', 'git', 'docker', 'aws',
        'interview', 'coding-interview', 'technical-interview'
    }
    return any(str(tag).lower() in interview_keywords for tag in tags_list if pd.notna(tag))

def process_archive_data(sample_size=10000, min_answer_score=5):
    """
    Process archive data to create training dataset
    
    Parameters:
    - sample_size: Number of question-answer pairs to extract
    - min_answer_score: Minimum score for an answer to be included
    """
    
    print("Loading archive data...")
    print("=" * 60)
    
    # Load data in chunks to handle large files
    print("\n1. Loading Questions...")
    questions_df = pd.read_csv(
        os.path.join(ARCHIVE_PATH, 'Questions.csv'),
        encoding='latin-1',
        nrows=sample_size * 5  # Read more to filter
    )
    print(f"   Loaded {len(questions_df):,} questions")
    
    print("\n2. Loading Answers...")
    answers_df = pd.read_csv(
        os.path.join(ARCHIVE_PATH, 'Answers.csv'),
        encoding='latin-1',
        nrows=sample_size * 10  # More answers than questions
    )
    print(f"   Loaded {len(answers_df):,} answers")
    
    print("\n3. Loading Tags...")
    tags_df = pd.read_csv(
        os.path.join(ARCHIVE_PATH, 'Tags.csv'),
        encoding='latin-1'
    )
    print(f"   Loaded {len(tags_df):,} tags")
    
    # Group tags by question ID
    print("\n4. Grouping tags by question...")
    question_tags = tags_df.groupby('Id')['Tag'].apply(list).to_dict()
    
    # Filter questions with relevant tags
    print("\n5. Filtering interview-relevant questions...")
    relevant_questions = []
    for idx, row in questions_df.iterrows():
        q_id = row['Id']
        if q_id in question_tags:
            tags = question_tags[q_id]
            if filter_interview_related(tags):
                relevant_questions.append(q_id)
    
    print(f"   Found {len(relevant_questions):,} relevant questions")
    
    # Filter to relevant questions only
    questions_df = questions_df[questions_df['Id'].isin(relevant_questions)]
    
    # Filter answers by score and parent question
    print("\n6. Filtering high-quality answers...")
    answers_df = answers_df[
        (answers_df['ParentId'].isin(relevant_questions)) &
        (answers_df['Score'] >= min_answer_score)
    ]
    print(f"   Found {len(answers_df):,} high-quality answers")
    
    # Create training data
    print("\n7. Creating training dataset...")
    training_data = []
    
    for idx, answer in answers_df.iterrows():
        parent_id = answer['ParentId']
        question = questions_df[questions_df['Id'] == parent_id]
        
        if len(question) == 0:
            continue
            
        question = question.iloc[0]
        
        # Clean text
        question_text = clean_html(question['Title'])
        answer_text = clean_html(answer['Body'])
        
        # Skip very short answers
        if len(answer_text) < 50:
            continue
        
        # Get tags
        tags = question_tags.get(parent_id, [])
        # Convert tags to strings and filter out NaN values
        tags = [str(tag) for tag in tags if pd.notna(tag)]
        
        # Normalize score
        score = normalize_score(answer['Score'])
        
        training_data.append({
            'question': question_text,
            'user_answer': answer_text,
            'score': score,
            'tags': ', '.join(tags),
            'original_score': answer['Score'],
            'question_id': parent_id,
            'answer_id': answer['Id']
        })
        
        if len(training_data) >= sample_size:
            break
    
    # Create DataFrame
    print(f"\n8. Creating final dataset...")
    final_df = pd.DataFrame(training_data)
    
    # Save to CSV
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    print(f"\n{'=' * 60}")
    print(f"✓ Successfully created training data!")
    print(f"{'=' * 60}")
    print(f"\nDataset Statistics:")
    print(f"  - Total records: {len(final_df):,}")
    print(f"  - Score range: {final_df['score'].min():.1f} - {final_df['score'].max():.1f}")
    print(f"  - Average score: {final_df['score'].mean():.1f}")
    print(f"  - Median score: {final_df['score'].median():.1f}")
    print(f"  - Unique questions: {final_df['question_id'].nunique():,}")
    print(f"\nScore distribution:")
    print(final_df['score'].value_counts().sort_index())
    print(f"\nTop tags:")
    all_tags = []
    for tags in final_df['tags']:
        all_tags.extend(tags.split(', '))
    from collections import Counter
    tag_counts = Counter(all_tags)
    for tag, count in tag_counts.most_common(15):
        print(f"  {tag}: {count}")
    
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"\nSample data:")
    print(final_df[['question', 'score', 'tags']].head(3))
    
    return final_df

if __name__ == "__main__":
    # Process with default parameters
    # Adjust sample_size and min_answer_score as needed
    df = process_archive_data(
        sample_size=10000,      # Number of Q&A pairs to extract
        min_answer_score=5       # Minimum SO score for quality answers
    )
