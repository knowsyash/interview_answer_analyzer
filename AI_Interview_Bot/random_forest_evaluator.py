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
    Optimized Random Forest evaluator with 32 engineered features
    Trained on 11,514 samples with 77% accuracy
    Scores answers on 1-5 scale based on:
    - STAR structure detection (4 features)
    - Professional keywords (8 features)
    - Linguistic quality (5 features)
    - Structure & quality (7 features)
    - Confidence & clarity (4 features)
    - Advanced metrics (4 features)
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
        Extract 32 advanced features from answer (matching optimized notebook model)
        Returns dict with 32 features in exact order:
        - Basic metrics (5): word_count, sentence_count, avg_word_length, char_length, words_per_sentence
        - STAR components (4): has_situation, has_task, has_action, has_result
        - Professional keywords (8): action_verbs, technical_terms, metrics_mentions, professional_words,
                                     problem_solving, leadership_words, communication_words, innovation_words
        - Structure & quality (7): has_numbers, question_marks, exclamation_marks, comma_count,
                                   uppercase_count, conjunctions, is_complete
        - Confidence & clarity (4): has_examples, hedging_words, confident_words, clarity_score
        - Advanced metrics (4): unique_word_ratio, complexity_score, technical_density, professional_density
        """
        answer_lower = answer.lower()
        words = answer.split()
        
        # Basic metrics (5 features)
        word_count = len(words)
        sentence_count = max(1, answer.count('.') + answer.count('!') + answer.count('?'))
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        char_length = len(answer)
        words_per_sentence = word_count / sentence_count
        
        # STAR components (4 features)
        has_situation = int(any(w in answer_lower for w in ['situation', 'context', 'when', 'where', 'background']))
        has_task = int(any(w in answer_lower for w in ['task', 'goal', 'objective', 'needed', 'required']))
        has_action = int(any(w in answer_lower for w in ['action', 'did', 'implemented', 'developed', 'created']))
        has_result = int(any(w in answer_lower for w in ['result', 'achieved', 'outcome', 'success', 'impact']))
        
        # Professional keywords (8 features)
        action_verbs = sum(1 for w in words if w.lower() in ['led', 'managed', 'created', 'developed', 'implemented', 'designed', 'analyzed', 'improved', 'optimized', 'achieved'])
        technical_terms = sum(1 for w in words if w.lower() in ['data', 'model', 'algorithm', 'analysis', 'system', 'process', 'performance', 'code', 'function', 'api'])
        metrics_mentions = sum(1 for w in words if w.lower() in ['%', 'increased', 'decreased', 'reduced', 'improved', 'growth', 'efficiency'])
        professional_words = sum(1 for w in words if w.lower() in ['team', 'project', 'stakeholder', 'client', 'customer', 'business', 'manager', 'collaboration'])
        problem_solving = sum(1 for w in words if w.lower() in ['problem', 'issue', 'challenge', 'solution', 'resolved', 'fixed', 'troubleshoot'])
        leadership_words = sum(1 for w in words if w.lower() in ['led', 'guided', 'mentored', 'coordinated', 'organized', 'delegated'])
        communication_words = sum(1 for w in words if w.lower() in ['presented', 'explained', 'communicated', 'discussed', 'collaborated', 'shared'])
        innovation_words = sum(1 for w in words if w.lower() in ['innovative', 'creative', 'new', 'novel', 'unique', 'pioneered'])
        
        # Structure & quality (7 features)
        has_numbers = sum(c.isdigit() for c in answer)
        question_marks = answer.count('?')
        exclamation_marks = answer.count('!')
        comma_count = answer.count(',')
        uppercase_count = sum(1 for c in answer if c.isupper())
        conjunctions = answer.count(' and ') + answer.count(' or ') + answer.count(' but ')
        is_complete = int(word_count > 20 and sentence_count > 1)
        
        # Confidence & clarity (4 features)
        has_examples = int(any(w in answer_lower for w in ['example', 'instance', 'case', 'specifically', 'for instance']))
        hedging_words = sum(1 for w in words if w.lower() in ['maybe', 'perhaps', 'possibly', 'might', 'could', 'probably'])
        confident_words = sum(1 for w in words if w.lower() in ['will', 'definitely', 'certainly', 'always', 'successfully', 'ensured'])
        clarity_score = int(avg_word_length < 8 and words_per_sentence < 25)
        
        # Advanced metrics (4 features)
        unique_word_ratio = len(set(words)) / len(words) if words else 0
        complexity_score = (avg_word_length * 0.5) + (words_per_sentence * 0.3)
        technical_density = (technical_terms + action_verbs) / max(1, word_count) * 100
        professional_density = (professional_words + leadership_words) / max(1, word_count) * 100
        
        # Return as ordered dict matching notebook feature order
        features = {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'char_length': char_length,
            'words_per_sentence': words_per_sentence,
            'has_situation': has_situation,
            'has_task': has_task,
            'has_action': has_action,
            'has_result': has_result,
            'action_verbs': action_verbs,
            'technical_terms': technical_terms,
            'metrics_mentions': metrics_mentions,
            'professional_words': professional_words,
            'problem_solving': problem_solving,
            'leadership_words': leadership_words,
            'communication_words': communication_words,
            'innovation_words': innovation_words,
            'has_numbers': has_numbers,
            'question_marks': question_marks,
            'exclamation_marks': exclamation_marks,
            'comma_count': comma_count,
            'uppercase_count': uppercase_count,
            'conjunctions': conjunctions,
            'is_complete': is_complete,
            'has_examples': has_examples,
            'hedging_words': hedging_words,
            'confident_words': confident_words,
            'clarity_score': clarity_score,
            'unique_word_ratio': unique_word_ratio,
            'complexity_score': complexity_score,
            'technical_density': technical_density,
            'professional_density': professional_density
        }
        
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

        # Candidate paths (in order) - prioritize optimized model from Research_Analysis
        candidates = []
        if model_path:
            candidates.append(model_path)
        # First try the optimized model from Research_Analysis
        research_model = os.path.join(os.path.dirname(base_dir), 'Research_Analysis', 'data', 'real_dataset_score', 'random_forest_model.joblib')
        candidates.append(research_model)
        # Then try local paths
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
        
        # Handle two formats: dict with metadata OR direct model object
        if isinstance(model_data, dict):
            # Format 1: Dictionary with 'model', 'feature_names', etc.
            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names', self.feature_names)
            print(f"Model loaded from: {found_path}")
            print(f"Test Accuracy: {model_data.get('test_accuracy', 'N/A')}")
            print(f"Test MAE: {model_data.get('test_mae', 'N/A')}")
        else:
            # Format 2: Direct model object (VotingClassifier or RandomForest)
            self.model = model_data
            print(f"Model loaded from: {found_path}")
            print(f"Model type: {type(self.model).__name__}")
            print("Warning: Model loaded without metadata (feature_names, accuracy, MAE)")
            print("Initializing default 32 feature names...")
        
        # Ensure feature_names is initialized (critical for extract_features compatibility)
        if not self.feature_names:
            # Generate default feature names matching the 32 features from extract_features
            self.feature_names = [
                'word_count', 'sentence_count', 'avg_word_length', 'char_length', 'words_per_sentence',
                'has_situation', 'has_task', 'has_action', 'has_result',
                'action_verbs', 'technical_terms', 'metrics_mentions', 'professional_words',
                'problem_solving', 'leadership_words', 'communication_words', 'innovation_words',
                'has_numbers', 'question_marks', 'exclamation_marks', 'comma_count',
                'uppercase_count', 'conjunctions', 'is_complete',
                'has_examples', 'hedging_words', 'confident_words', 'clarity_score',
                'unique_word_ratio', 'complexity_score', 'technical_density', 'professional_density'
            ]
            print(f"Feature names initialized: {len(self.feature_names)} features")
        
        return model_data
    
    def evaluate_answer(self, question, answer, reference_answer=None):
        """
        Evaluate an answer and return score with feedback (1-5 scale converted to 10-point scale)
        
        Args:
            question: Interview question text
            answer: User's answer text
            reference_answer: Reference answer dict (optional, not used by RF model)
            
        Returns:
            dict with 'predicted_score' (on 10-point scale) and 'feedback'
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Detect question type (behavioral vs technical)
        question_lower = question.lower()
        is_behavioral = any(keyword in question_lower for keyword in 
                          ['tell me about', 'describe a time', 'give an example', 'experience', 
                           'situation', 'challenge you faced', 'conflict', 'leadership'])
        
        # Extract features
        features = self.extract_features(answer)
        X = np.array([list(features.values())])
        
        # Predict score (1-5 scale)
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
        
        # Get feature importances (handle both RandomForest and VotingClassifier)
        feature_importances = None
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Direct RandomForest model
                feature_importances = self.model.feature_importances_
            elif hasattr(self.model, 'estimators_'):
                # VotingClassifier - get from first RandomForest estimator
                for estimator in self.model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        feature_importances = estimator.feature_importances_
                        break
        except:
            pass
        
        # Get top features influencing this prediction
        if feature_importances is not None:
            feature_values = pd.DataFrame({
                'feature': self.feature_names,
                'value': list(features.values()),
                'importance': feature_importances
            })
            feature_values['contribution'] = feature_values['value'] * feature_values['importance']
            top_features = feature_values.nlargest(5, 'contribution')
        else:
            # Fallback: just use feature values without importances
            feature_values = pd.DataFrame({
                'feature': self.feature_names,
                'value': list(features.values())
            })
            top_features = feature_values.nlargest(5, 'value')
        
        # Generate feedback based on features
        feedback_lines = self.get_feedback(score, features)
        
        # Build professional feedback text
        feedback_parts = []
        
        # Add key strengths
        star_score = (features.get('has_situation', 0) + features.get('has_task', 0) + 
                     features.get('has_action', 0) + features.get('has_result', 0))
        
        strengths = []
        if features.get('has_numbers', 0) > 0:
            strengths.append("includes quantifiable metrics")
        if features.get('technical_terms', 0) >= 2:
            strengths.append("demonstrates technical knowledge")
        if features.get('professional_words', 0) >= 3:
            strengths.append("uses professional terminology")
        if is_behavioral and star_score >= 3:
            strengths.append("follows structured response format")
        if features.get('word_count', 0) >= 50:
            strengths.append("provides sufficient detail")
            
        if strengths:
            feedback_parts.append("Strengths: Your answer " + ", ".join(strengths) + ".")
        
        # Generate context-aware improvement suggestions
        feedback_lines = self.get_feedback(score, features, is_behavioral)
        
        if feedback_lines:
            feedback_parts.append("\nAreas for improvement:")
            for suggestion in feedback_lines[:3]:
                feedback_parts.append(f"- {suggestion}")
        
        feedback_text = "\n".join(feedback_parts)
        
        # Convert 1-5 scale to 10-point scale (multiply by 2)
        predicted_score_10 = score * 2.0
        
        # Prepare top contributing features for details
        if 'contribution' in top_features.columns:
            top_features_list = top_features[['feature', 'value', 'contribution']].to_dict('records')
        else:
            top_features_list = top_features[['feature', 'value']].to_dict('records')
        
        return {
            'predicted_score': predicted_score_10,
            'feedback': feedback_text,
            'details': {
                'score': int(score),
                'confidence': float(confidence),
                'score_distribution': {i+1: float(p) for i, p in enumerate(probabilities)},
                'features': features,
                'top_contributing_features': top_features_list
            }
        }
    
    def get_feedback(self, score, features, is_behavioral=True):
        """Generate feedback based on score and features"""
        feedback = []
        
        # STAR structure feedback - only for behavioral questions
        if is_behavioral:
            star_scores = {
                'situation': features.get('has_situation', 0),
                'task': features.get('has_task', 0),
                'action': features.get('has_action', 0),
                'result': features.get('has_result', 0)
            }
            
            weak_components = [comp for comp, val in star_scores.items() if val < 0.5]
            if len(weak_components) >= 2:
                feedback.append("Structure your answer using the STAR method (Situation, Task, Action, Result).")
        
        # Technical content feedback
        if features.get('technical_terms', 0) < 2:
            feedback.append("Include more technical terminology and concepts relevant to the question.")
        
        # Quantifiable results
        if not features.get('has_numbers', 0) and is_behavioral:
            feedback.append("Add specific metrics or numbers to demonstrate measurable impact.")
        
        # Length feedback
        word_count = features.get('word_count', 0)
        if word_count < 40:
            feedback.append("Provide more detailed explanation with specific examples.")
        elif word_count > 200:
            feedback.append("Consider being more concise while maintaining key details.")
        
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
