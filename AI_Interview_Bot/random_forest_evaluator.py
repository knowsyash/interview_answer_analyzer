"""
Random Forest Answer Evaluator
Advanced ML-based scoring using Random Forest with engineered features
Trained on 1,514 behavioral + web dev interview Q&A pairs
"""

import numpy as np
import pandas as pd
import re
import os
import json
import joblib
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else 
                      f'corpora/{resource}' if resource in ['stopwords', 'wordnet'] else 
                      f'taggers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)


class RandomForestAnswerEvaluator:
    """
    Advanced Random Forest evaluator with 23 engineered features
    Scores answers on 1-5 scale based on:
    - STAR structure detection
    - Competency keywords
    - Linguistic quality
    - Domain-specific terms
    """
    
    def __init__(self):
        self.model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Define STAR keywords (140 total across 4 components)
        self.star_keywords = {
            'situation': [
                'faced', 'encountered', 'situation', 'problem', 'challenge', 'scenario',
                'when', 'during', 'time', 'experience', 'context', 'background', 'circumstances',
                'environment', 'setting', 'position', 'working', 'project', 'assignment', 'case',
                'instance', 'occasion', 'period', 'phase', 'stage', 'moment', 'point', 'dealing',
                'handling', 'managing', 'overseeing', 'involved', 'part', 'role', 'capacity'
            ],  # 35 keywords
            
            'task': [
                'responsible', 'assigned', 'task', 'objective', 'goal', 'required', 'needed',
                'role', 'duty', 'expected', 'mission', 'job', 'responsibility', 'accountable',
                'charge', 'obligation', 'mandate', 'brief', 'directive', 'instruction', 'order',
                'requirement', 'specification', 'target', 'aim', 'purpose', 'intention', 'plan',
                'agenda', 'scope', 'deliverable', 'milestone', 'deadline', 'timeline', 'priority'
            ],  # 35 keywords
            
            'action': [
                'did', 'implemented', 'executed', 'performed', 'managed', 'led', 'organized',
                'created', 'developed', 'improved', 'coordinated', 'initiated', 'established',
                'designed', 'built', 'analyzed', 'investigated', 'researched', 'planned',
                'strategized', 'facilitated', 'conducted', 'oversaw', 'supervised', 'directed',
                'guided', 'mentored', 'trained', 'coached', 'collaborated', 'communicated',
                'presented', 'negotiated', 'resolved', 'solved', 'fixed', 'addressed', 'handled',
                'tackled', 'approached', 'utilized', 'applied', 'leveraged', 'deployed', 'launched'
            ],  # 45 keywords
            
            'result': [
                'achieved', 'resulted', 'outcome', 'impact', 'success', 'improved', 'increased',
                'decreased', 'delivered', 'completed', 'exceeded', 'accomplished', 'attained',
                'reached', 'obtained', 'gained', 'secured', 'generated', 'produced', 'yielded',
                'realized', 'fulfilled', 'satisfied', 'met', 'surpassed', 'outperformed',
                'enhanced', 'optimized', 'streamlined', 'reduced', 'saved', 'earned', 'won',
                'captured', 'grew', 'expanded', 'boosted', 'maximized', 'minimized', 'efficiency',
                'productivity', 'quality', 'satisfaction', 'revenue', 'profit', 'performance',
                'metrics', 'kpi', 'measure', 'quantifiable'
            ]  # 50 keywords (total = 165, but using 140 key ones)
        }
        
        # Competency keywords (7 categories)
        self.competency_keywords = {
            'leadership': [
                'lead', 'led', 'leadership', 'manage', 'manager', 'direct', 'guide', 'mentor',
                'supervise', 'oversee', 'coordinate', 'delegate', 'empower', 'inspire', 'vision',
                'strategic', 'initiative', 'decision', 'authority', 'responsibility'
            ],
            'teamwork': [
                'team', 'collaborate', 'cooperation', 'together', 'group', 'colleagues', 'peers',
                'partner', 'coordinate', 'support', 'help', 'assist', 'contribute', 'share',
                'communication', 'participate', 'collective', 'joint', 'unified', 'synergy'
            ],
            'problem_solving': [
                'problem', 'solve', 'solution', 'issue', 'challenge', 'troubleshoot', 'debug',
                'analyze', 'investigate', 'research', 'identify', 'diagnose', 'resolve', 'fix',
                'address', 'overcome', 'creative', 'innovative', 'critical', 'analytical'
            ],
            'communication': [
                'communicate', 'present', 'explain', 'discuss', 'negotiate', 'persuade',
                'articulate', 'convey', 'express', 'clarify', 'inform', 'update', 'report',
                'brief', 'meeting', 'presentation', 'email', 'document', 'verbal', 'written'
            ],
            'technical': [
                'technical', 'technology', 'system', 'software', 'code', 'develop', 'program',
                'database', 'server', 'api', 'algorithm', 'framework', 'tool', 'platform',
                'architecture', 'design', 'implement', 'deploy', 'test', 'optimize'
            ],
            'result_orientation': [
                'achieve', 'accomplish', 'deliver', 'complete', 'success', 'goal', 'target',
                'objective', 'outcome', 'result', 'impact', 'performance', 'metric', 'kpi',
                'exceed', 'improve', 'increase', 'optimize', 'efficiency', 'productivity'
            ],
            'adaptability': [
                'adapt', 'flexible', 'change', 'transition', 'adjust', 'evolve', 'learn',
                'new', 'different', 'challenge', 'uncertain', 'dynamic', 'agile', 'pivot',
                'resilient', 'respond', 'handle', 'cope', 'manage', 'versatile'
            ]
        }
        
        self.feature_names = []
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, answer, reference_answer=""):
        """
        Extract 23 features from answer:
        1-4: STAR component detection (4 features)
        5-11: Competency keyword counts (7 features)
        12-16: Linguistic features (5 features)
        17-20: Structure features (4 features)
        21-23: Domain-specific features (3 features)
        """
        features = {}
        answer_clean = self.preprocess_text(answer)
        answer_lower = answer_clean.lower()
        tokens = word_tokenize(answer_lower)
        
        # Filter stop words for content analysis
        content_tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        
        # ===== STAR FEATURES (4 features) =====
        for component, keywords in self.star_keywords.items():
            count = sum(1 for keyword in keywords if keyword in answer_lower)
            features[f'star_{component}_count'] = min(count / 5, 1.0)  # Normalize to [0,1]
        
        # ===== COMPETENCY FEATURES (7 features) =====
        for competency, keywords in self.competency_keywords.items():
            count = sum(1 for keyword in keywords if keyword in answer_lower)
            features[f'competency_{competency}'] = min(count / 3, 1.0)  # Normalize to [0,1]
        
        # ===== LINGUISTIC FEATURES (5 features) =====
        # Length-based features
        word_count = len(tokens)
        features['word_count_normalized'] = min(word_count / 200, 1.0)  # Normalize to [0,1]
        features['sentence_count'] = min(len(re.split(r'[.!?]+', answer_clean)) / 10, 1.0)
        features['avg_word_length'] = np.mean([len(w) for w in content_tokens]) / 10 if content_tokens else 0
        
        # Vocabulary diversity (unique words / total words)
        features['vocabulary_diversity'] = len(set(content_tokens)) / len(content_tokens) if content_tokens else 0
        
        # Past tense verb usage (common in STAR responses)
        past_tense_verbs = ['did', 'managed', 'led', 'created', 'implemented', 'achieved', 
                           'developed', 'improved', 'coordinated', 'organized', 'completed']
        features['past_tense_usage'] = sum(1 for verb in past_tense_verbs if verb in answer_lower) / 5
        
        # ===== STRUCTURE FEATURES (4 features) =====
        # Presence of quantifiable results (numbers, percentages)
        features['has_numbers'] = 1.0 if re.search(r'\d+', answer) else 0.0
        features['has_percentage'] = 1.0 if re.search(r'\d+%', answer) else 0.0
        
        # First-person narrative (I, my, we)
        first_person = ['i ', 'my ', 'we ', 'our ', 'me ']
        features['first_person_usage'] = min(sum(1 for fp in first_person if fp in f' {answer_lower} ') / 5, 1.0)
        
        # Answer coherence (transition words)
        transition_words = ['first', 'then', 'next', 'after', 'finally', 'as a result', 
                          'therefore', 'however', 'additionally', 'furthermore']
        features['transition_words'] = min(sum(1 for tw in transition_words if tw in answer_lower) / 3, 1.0)
        
        # ===== DOMAIN-SPECIFIC FEATURES (3 features) =====
        # Web development keywords (for webdev questions)
        webdev_keywords = ['html', 'css', 'javascript', 'react', 'api', 'database', 'frontend', 
                          'backend', 'web', 'server', 'client', 'framework', 'library']
        features['webdev_relevance'] = min(sum(1 for kw in webdev_keywords if kw in answer_lower) / 5, 1.0)
        
        # Professional terminology
        professional_terms = ['stakeholder', 'deliverable', 'milestone', 'timeline', 'budget', 
                             'scope', 'requirement', 'specification', 'client', 'customer']
        features['professional_terms'] = min(sum(1 for pt in professional_terms if pt in answer_lower) / 3, 1.0)
        
        # Action-oriented language (strong verbs)
        action_verbs = ['executed', 'delivered', 'optimized', 'streamlined', 'spearheaded', 
                       'pioneered', 'orchestrated', 'facilitated', 'championed', 'transformed']
        features['action_oriented'] = min(sum(1 for av in action_verbs if av in answer_lower) / 3, 1.0)
        
        return features
    
    def prepare_training_data(self, csv_path):
        """Load and prepare training data from CSV"""
        print(f"Loading training data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Extract features for all answers
        X_list = []
        y_list = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(df)} samples...")
            
            answer = str(row.get('answer', ''))
            score = row.get('human_score', row.get('competency_score', row.get('score', 3)))  # Use human_score
            
            # Normalize scores to 1-5 range (webdev dataset has 1-10 scores)
            if score > 5:
                score = min(5, int((score / 10.0) * 5 + 0.5))  # Scale 10-point to 5-point
            
            if answer and len(answer.strip()) > 10:
                features = self.extract_features(answer)
                X_list.append(list(features.values()))
                y_list.append(int(score))
        
        # Store feature names
        if X_list:
            sample_features = self.extract_features(df.iloc[0]['answer'])
            self.feature_names = list(sample_features.keys())
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Feature names ({len(self.feature_names)}): {self.feature_names}")
        print(f"Target distribution: {Counter(y)}")
        
        return X, y
    
    def train_model(self, data_dir=None, save_path=None):
        """
        Train Random Forest model on interview dataset
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'real_dataset_score')
        
        # Load both datasets
        behavioral_csv = os.path.join(data_dir, 'interview_data_with_scores.csv')
        webdev_csv = os.path.join(data_dir, 'webdev_interview_qa.csv')
        
        print("=" * 60)
        print("TRAINING RANDOM FOREST ANSWER EVALUATOR")
        print("=" * 60)
        
        # Prepare data from both sources
        X_behavioral, y_behavioral = self.prepare_training_data(behavioral_csv)
        X_webdev, y_webdev = self.prepare_training_data(webdev_csv)
        
        # Combine datasets
        X = np.vstack([X_behavioral, X_webdev])
        y = np.concatenate([y_behavioral, y_webdev])
        
        print(f"\nCombined dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Score distribution: {Counter(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train Random Forest
        print("\nTraining Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=200,          # Number of trees
            max_depth=20,              # Maximum depth
            min_samples_split=5,       # Minimum samples to split
            min_samples_leaf=2,        # Minimum samples per leaf
            max_features='sqrt',       # Features per split
            random_state=42,
            n_jobs=-1,                 # Use all cores
            verbose=1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Training performance
        y_train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        # Test performance
        y_test_pred = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"\nTraining Accuracy: {train_acc:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Within ±1 score accuracy
        within_1 = np.mean(np.abs(y_test - y_test_pred) <= 1)
        print(f"Within ±1 Score: {within_1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Save model
        if save_path is None:
            save_path = os.path.join(data_dir, 'random_forest_model.joblib')
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'test_accuracy': test_acc,
            'test_mae': test_mae
        }
        
        joblib.dump(model_data, save_path)
        print(f"\nModel saved to: {save_path}")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'within_1_accuracy': within_1,
            'feature_importance': feature_importance
        }
    
    def load_model(self, model_path=None):
        """Load trained Random Forest model.

        Tries several common locations to be robust to different data folder layouts.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Candidate paths (in order)
        candidates = []
        if model_path:
            candidates.append(model_path)
        candidates.append(os.path.join(base_dir, 'data', 'random_forest_model.joblib'))
        candidates.append(os.path.join(base_dir, 'data', 'real_dataset_score', 'random_forest_model.joblib'))
        candidates.append(os.path.join(base_dir, 'data', 'real_use', 'random_forest_model.joblib'))

        found_path = None
        for p in candidates:
            if p and os.path.exists(p):
                found_path = p
                break

        if not found_path:
            raise FileNotFoundError(
                f"Model not found. Checked paths: {candidates}"
            )

        model_data = joblib.load(found_path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from: {found_path}")
        print(f"Test Accuracy: {model_data.get('test_accuracy', 'N/A'):.4f}")
        print(f"Test MAE: {model_data.get('test_mae', 'N/A'):.4f}")
        
        return model_data
    
    def evaluate_answer(self, answer, question="", return_details=False):
        """
        Evaluate an answer and return score (1-5 scale)
        
        Args:
            answer: User's answer text
            question: Interview question (optional, for context)
            return_details: If True, return detailed feature breakdown
            
        Returns:
            If return_details=False: score (1-5)
            If return_details=True: dict with score, confidence, features
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Extract features
        features = self.extract_features(answer)
        X = np.array([list(features.values())])
        
        # Predict score
        score = self.model.predict(X)[0]
        
        # Get prediction probabilities for confidence
        probabilities = self.model.predict_proba(X)[0]
        
        # Get class labels from the model
        class_labels = self.model.classes_
        
        # Find index of predicted score in class labels
        try:
            score_index = list(class_labels).index(score)
            confidence = probabilities[score_index]
        except (ValueError, IndexError):
            confidence = max(probabilities)  # Fallback to highest probability
        
        if return_details:
            # Get top features influencing this prediction
            feature_values = pd.DataFrame({
                'feature': self.feature_names,
                'value': list(features.values()),
                'importance': self.model.feature_importances_
            })
            feature_values['contribution'] = feature_values['value'] * feature_values['importance']
            top_features = feature_values.nlargest(5, 'contribution')
            
            return {
                'score': int(score),
                'confidence': float(confidence),
                'score_distribution': {i+1: float(p) for i, p in enumerate(probabilities)},
                'features': features,
                'top_contributing_features': top_features[['feature', 'value', 'contribution']].to_dict('records')
            }
        
        return int(score)
    
    def get_feedback(self, score, features):
        """Generate feedback based on score and features"""
        feedback = []
        
        # STAR structure feedback
        star_scores = {
            'situation': features.get('star_situation_count', 0),
            'task': features.get('star_task_count', 0),
            'action': features.get('star_action_count', 0),
            'result': features.get('star_result_count', 0)
        }
        
        weak_components = [comp for comp, val in star_scores.items() if val < 0.3]
        if weak_components:
            feedback.append(f"Consider adding more {', '.join(weak_components).upper()} details to strengthen your STAR structure.")
        
        # Quantifiable results
        if not features.get('has_numbers', 0):
            feedback.append("Add specific metrics or numbers to demonstrate measurable impact.")
        
        # Length feedback
        if features.get('word_count_normalized', 0) < 0.3:
            feedback.append("Provide more detailed explanation - aim for 60-150 words.")
        elif features.get('word_count_normalized', 0) > 0.9:
            feedback.append("Consider being more concise - focus on key points.")
        
        # Action-oriented language
        if features.get('action_oriented', 0) < 0.2:
            feedback.append("Use stronger action verbs (executed, delivered, optimized, etc.).")
        
        return feedback


# CLI for training
if __name__ == "__main__":
    import sys
    
    evaluator = RandomForestAnswerEvaluator()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Train model
        results = evaluator.train_model()
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Final Test MAE: {results['test_mae']:.4f}")
        print(f"Within ±1 Score: {results['within_1_accuracy']:.4f}")
        
    else:
        print("Random Forest Answer Evaluator")
        print("\nUsage:")
        print("  python random_forest_evaluator.py train    # Train new model")
        print("\nOr import and use in your code:")
        print("  from random_forest_evaluator import RandomForestAnswerEvaluator")
        print("  evaluator = RandomForestAnswerEvaluator()")
        print("  evaluator.load_model()")
        print("  score = evaluator.evaluate_answer('Your answer here')")
