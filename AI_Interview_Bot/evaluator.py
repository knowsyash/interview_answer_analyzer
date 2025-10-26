from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import string
import re
import json
import os

class AnswerEvaluator:
    def __init__(self):
        """Initialize the evaluator with required resources."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: NLTK data download failed: {str(e)}")
            self.stop_words = set()
        
        # Initialize vectorizer and weights
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.keyword_weight = 0.4
        self.semantic_weight = 0.4
        self.structure_weight = 0.2
        
        # Load human evaluation data
        self.human_data = self.load_human_evaluations()

    def load_human_evaluations(self):
        """Load human evaluation data for comparison."""
        try:
            data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            eval_path = os.path.join(data_dir, "data", "human_evaluations.json")
            
            if os.path.exists(eval_path):
                with open(eval_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {"evaluations": []}
        except Exception as e:
            print(f"Warning: Could not load human evaluations: {str(e)}")
            return {"evaluations": []}

    def clean_text(self, text):
        """Clean and preprocess text for evaluation."""
        try:
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Warning: Text cleaning failed: {str(e)}")
            return text

    def calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using TF-IDF and cosine similarity."""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Clean texts
            text1_clean = self.clean_text(text1)
            text2_clean = self.clean_text(text2)
            
            # Calculate TF-IDF and similarity
            tfidf_matrix = self.vectorizer.fit_transform([text1_clean, text2_clean])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            return float(similarity[0][0])
        except Exception as e:
            print(f"Warning: Similarity calculation failed: {str(e)}")
            return 0.0

    def evaluate_structure(self, answer):
        """Evaluate the structure and completeness of an answer."""
        try:
            # Clean the answer
            clean_answer = self.clean_text(answer)
            words = clean_answer.split()
            
            if not words:
                return 0.0
            
            scores = []
            
            # Length score
            length = len(words)
            if length < 5:
                scores.append(0.2)
            elif length < 10:
                scores.append(0.4)
            elif length < 50:
                scores.append(0.8)
            elif length < 100:
                scores.append(1.0)
            else:
                scores.append(0.6)
            
            # Sentence structure
            sentences = [s for s in answer.split('.') if s.strip()]
            scores.append(min(1.0, len(sentences) / 3))
            
            # Vocabulary diversity
            unique_words = len(set(words))
            diversity = unique_words / len(words) if words else 0
            scores.append(min(1.0, diversity * 2))
            
            return np.mean(scores)
        except Exception as e:
            print(f"Warning: Structure evaluation failed: {str(e)}")
            return 0.5

    def compare_with_human_evaluation(self, role, question, answer):
        """Compare machine evaluation with human evaluation data."""
        try:
            # Find matching question in human evaluations
            for eval_data in self.human_data["evaluations"]:
                if eval_data["role"] == role and eval_data["question"].lower() == question.lower():
                    human_scores = []
                    machine_scores = []
                    
                    # Calculate scores for each response
                    for response in eval_data["responses"]:
                        human_scores.append(response["human_score"] / 10.0)  # Normalize to 0-1
                        machine_score, _ = self.evaluate_answer(
                            response["answer"],
                            answer,
                            response.get("keywords", [])
                        )
                        machine_scores.append(machine_score)
                    
                    if human_scores and machine_scores:
                        correlation = np.corrcoef(human_scores, machine_scores)[0, 1]
                        mae = np.mean(np.abs(np.array(human_scores) - np.array(machine_scores)))
                        return {
                            "correlation": correlation,
                            "mean_absolute_error": mae,
                            "human_mean": np.mean(human_scores),
                            "machine_mean": np.mean(machine_scores)
                        }
            
            return None
        except Exception as e:
            print(f"Warning: Human evaluation comparison failed: {str(e)}")
            return None

    def evaluate_answer(self, user_answer, correct_answer, keywords=None):
        """
        Evaluate an answer comprehensively and return detailed feedback.
        
        Args:
            user_answer (str): The answer provided by the user
            correct_answer (str): The reference answer to compare against (can be empty)
            keywords (list): Optional list of important keywords to look for
            
        Returns:
            tuple: (score, feedback_dict)
        """
        try:
            # Calculate semantic similarity (only if we have a reference answer)
            if correct_answer and correct_answer.strip():
                semantic_score = self.calculate_semantic_similarity(user_answer, correct_answer)
            else:
                # If no reference answer, evaluate based on structure and length only
                semantic_score = self.evaluate_structure(user_answer)
            
            # Calculate structural score
            structure_score = self.evaluate_structure(user_answer)
            
            # Calculate keyword score if keywords provided
            if keywords:
                user_words = set(self.clean_text(user_answer).split())
                keyword_matches = sum(1 for k in keywords if k.lower() in user_words)
                keyword_score = keyword_matches / len(keywords)
            else:
                keyword_score = semantic_score  # Use semantic score if no keywords
            
            # Calculate final score
            final_score = (
                self.semantic_weight * semantic_score +
                self.structure_weight * structure_score +
                self.keyword_weight * keyword_score
            )
            
            # Prepare feedback
            feedback = {
                "semantic_score": round(semantic_score, 2),
                "structure_score": round(structure_score, 2),
                "keyword_score": round(keyword_score, 2),
                "final_score": round(final_score, 2),
                "improvements": []
            }
            
            # Add improvement suggestions
            if semantic_score < 0.6 and correct_answer:
                feedback["improvements"].append(
                    "Your answer could be more closely aligned with the expected response."
                )
            if structure_score < 0.6:
                feedback["improvements"].append(
                    "Consider providing a more structured and complete response."
                )
            if keyword_score < 0.6:
                feedback["improvements"].append(
                    "Try to include more relevant technical terms and keywords."
                )
            
            return final_score, feedback
            
        except Exception as e:
            print(f"Warning: Answer evaluation failed: {str(e)}")
            return 0.5, {"error": str(e)}

# Legacy function for backward compatibility
def evaluate_answer(user_answer, correct_answer):
    """Legacy function for backward compatibility."""
    evaluator = AnswerEvaluator()
    score, _ = evaluator.evaluate_answer(user_answer, correct_answer)
    return score
