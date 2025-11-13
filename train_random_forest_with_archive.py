"""
Train Random Forest Model with Stack Overflow Archive Data

This script trains the Random Forest scoring model using the processed
Stack Overflow data from the archive folder.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Paths
DATA_PATH = r"AI_Interview_Bot\data"
STACKOVERFLOW_FILE = os.path.join(DATA_PATH, "stackoverflow_training_data.csv")
EXISTING_DATA_FILE = os.path.join(DATA_PATH, "interview_data_with_scores.csv")
COMBINED_OUTPUT_FILE = os.path.join(DATA_PATH, "combined_training_data.csv")
MODEL_OUTPUT_FILE = os.path.join(DATA_PATH, "random_forest_model.joblib")
VECTORIZER_OUTPUT_FILE = os.path.join(DATA_PATH, "tfidf_vectorizer.joblib")

def extract_features(df):
    """
    Extract features from question-answer pairs for Random Forest training
    """
    features = []
    
    for idx, row in df.iterrows():
        question = str(row['question'])
        answer = str(row['user_answer'])
        
        # Text-based features
        answer_length = len(answer)
        answer_word_count = len(answer.split())
        
        # Question-answer similarity (simple word overlap)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        word_overlap = len(question_words & answer_words) / max(len(question_words), 1)
        
        # Technical indicators
        has_code = 1 if any(word in answer.lower() for word in ['function', 'class', 'return', 'def', 'var', 'const']) else 0
        has_example = 1 if any(word in answer.lower() for word in ['example', 'for instance', 'such as']) else 0
        
        # Complexity indicators
        avg_word_length = np.mean([len(word) for word in answer.split()]) if answer_word_count > 0 else 0
        sentence_count = answer.count('.') + answer.count('!') + answer.count('?')
        
        features.append([
            answer_length,
            answer_word_count,
            word_overlap,
            has_code,
            has_example,
            avg_word_length,
            sentence_count
        ])
    
    feature_names = [
        'answer_length',
        'answer_word_count',
        'word_overlap',
        'has_code',
        'has_example',
        'avg_word_length',
        'sentence_count'
    ]
    
    return np.array(features), feature_names

def add_tfidf_features(df, max_features=100):
    """
    Add TF-IDF features to the dataset
    """
    # Combine question and answer for context
    combined_text = df['question'] + " " + df['user_answer']
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform
    tfidf_features = vectorizer.fit_transform(combined_text).toarray()
    
    return tfidf_features, vectorizer

def train_model(use_tfidf=True, combine_with_existing=True):
    """
    Train Random Forest model with archive data
    
    Parameters:
    - use_tfidf: Whether to include TF-IDF features
    - combine_with_existing: Whether to combine with existing interview data
    """
    
    print("Training Random Forest Model with Archive Data")
    print("=" * 60)
    
    # Load Stack Overflow data
    print("\n1. Loading Stack Overflow data...")
    so_df = pd.read_csv(STACKOVERFLOW_FILE)
    print(f"   Loaded {len(so_df):,} Stack Overflow Q&A pairs")
    
    # Optionally combine with existing data
    if combine_with_existing and os.path.exists(EXISTING_DATA_FILE):
        print("\n2. Loading existing interview data...")
        existing_df = pd.read_csv(EXISTING_DATA_FILE)
        print(f"   Loaded {len(existing_df):,} existing interview Q&A pairs")
        
        # Rename columns to match Stack Overflow data
        existing_df = existing_df.rename(columns={
            'answer': 'user_answer',
            'human_score': 'score'
        })
        
        # Ensure columns match
        existing_df = existing_df[['question', 'user_answer', 'score']]
        so_df_subset = so_df[['question', 'user_answer', 'score']]
        
        # Combine datasets
        combined_df = pd.concat([so_df_subset, existing_df], ignore_index=True)
        print(f"\n   Combined dataset: {len(combined_df):,} total pairs")
        
        # Save combined data
        combined_df.to_csv(COMBINED_OUTPUT_FILE, index=False)
        print(f"   Saved to: {COMBINED_OUTPUT_FILE}")
        
        training_df = combined_df
    else:
        training_df = so_df[['question', 'user_answer', 'score']]
    
    # Extract features
    print("\n3. Extracting features...")
    basic_features, feature_names = extract_features(training_df)
    print(f"   Extracted {basic_features.shape[1]} basic features")
    
    # Add TF-IDF features if requested
    if use_tfidf:
        print("\n4. Computing TF-IDF features...")
        tfidf_features, vectorizer = add_tfidf_features(training_df, max_features=100)
        print(f"   Extracted {tfidf_features.shape[1]} TF-IDF features")
        
        # Combine features
        X = np.hstack([basic_features, tfidf_features])
        all_feature_names = feature_names + [f"tfidf_{i}" for i in range(tfidf_features.shape[1])]
        
        # Save vectorizer
        joblib.dump(vectorizer, VECTORIZER_OUTPUT_FILE)
        print(f"   Saved TF-IDF vectorizer to: {VECTORIZER_OUTPUT_FILE}")
    else:
        X = basic_features
        all_feature_names = feature_names
    
    # Target variable
    y = training_df['score'].values
    
    print(f"\n5. Preparing training data...")
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")
    
    # Train Random Forest
    print("\n6. Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    print("   ✓ Training complete!")
    
    # Evaluate model
    print("\n7. Evaluating model...")
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n   Training Metrics:")
    print(f"     RMSE: {train_rmse:.3f}")
    print(f"     MAE:  {train_mae:.3f}")
    print(f"     R²:   {train_r2:.3f}")
    
    print(f"\n   Test Metrics:")
    print(f"     RMSE: {test_rmse:.3f}")
    print(f"     MAE:  {test_mae:.3f}")
    print(f"     R²:   {test_r2:.3f}")
    
    # Feature importance
    print("\n8. Feature Importance (Top 10):")
    feature_importance = sorted(
        zip(all_feature_names, rf_model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    for fname, importance in feature_importance:
        print(f"   {fname:25s}: {importance:.4f}")
    
    # Save model
    print("\n9. Saving model...")
    joblib.dump(rf_model, MODEL_OUTPUT_FILE)
    print(f"   ✓ Model saved to: {MODEL_OUTPUT_FILE}")
    
    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print("=" * 60)
    
    return rf_model, vectorizer if use_tfidf else None

if __name__ == "__main__":
    # Train the model
    model, vectorizer = train_model(
        use_tfidf=True,
        combine_with_existing=True
    )
