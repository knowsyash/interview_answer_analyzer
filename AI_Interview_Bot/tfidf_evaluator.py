"""
TF-IDF Based Answer Evaluator
Uses TF-IDF vectorization and cosine similarity for semantic answer evaluation
With NLTK preprocessing for better tokenization and lemmatization
"""

import re
import math
from collections import Counter
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only runs once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


class TFIDFAnswerEvaluator:
    """Evaluates answers using TF-IDF vectorization and cosine similarity"""
    
    def __init__(self):
        # Use NLTK's comprehensive stop words list
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize lemmatizer for word normalization
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        """
        Preprocess text using NLTK: tokenize, lowercase, lemmatize, remove stop words
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            list: List of cleaned and lemmatized tokens
        """
        # Lowercase
        text = text.lower()
        
        # Tokenize using NLTK (better than simple split)
        tokens = word_tokenize(text)
        
        # Remove punctuation, keep only alphanumeric tokens
        tokens = [token for token in tokens if token.isalnum()]
        
        # Remove stop words and very short tokens (less than 3 characters)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize tokens (convert to base form: running -> run, better -> good)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def compute_tf(self, tokens):
        """
        Compute Term Frequency (TF) for tokens
        TF(t) = (Number of times term t appears in document) / (Total number of terms in document)
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            dict: Term frequencies
        """
        tf_dict = {}
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return tf_dict
        
        token_counts = Counter(tokens)
        
        for token, count in token_counts.items():
            tf_dict[token] = count / total_tokens
            
        return tf_dict
    
    def compute_idf(self, documents):
        """
        Compute Inverse Document Frequency (IDF) for all terms across documents
        IDF(t) = log(Total number of documents / Number of documents containing term t)
        
        Args:
            documents (list): List of token lists
            
        Returns:
            dict: IDF scores for each unique term
        """
        idf_dict = {}
        total_docs = len(documents)
        
        if total_docs == 0:
            return idf_dict
        
        # Get all unique terms
        all_terms = set()
        for doc in documents:
            all_terms.update(doc)
        
        # Calculate IDF for each term
        for term in all_terms:
            docs_containing_term = sum(1 for doc in documents if term in doc)
            idf_dict[term] = math.log(total_docs / (1 + docs_containing_term))
        
        return idf_dict
    
    def compute_tfidf(self, tf_dict, idf_dict):
        """
        Compute TF-IDF scores
        TF-IDF(t) = TF(t) × IDF(t)
        
        Args:
            tf_dict (dict): Term frequencies
            idf_dict (dict): IDF scores
            
        Returns:
            dict: TF-IDF scores
        """
        tfidf_dict = {}
        
        for term, tf_value in tf_dict.items():
            idf_value = idf_dict.get(term, 0)
            tfidf_dict[term] = tf_value * idf_value
        
        return tfidf_dict
    
    def cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two TF-IDF vectors
        Cosine Similarity = (A · B) / (||A|| × ||B||)
        
        Args:
            vec1 (dict): First TF-IDF vector
            vec2 (dict): Second TF-IDF vector
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Get all unique terms from both vectors
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        if not all_terms:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in all_terms)
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def compute_keyword_overlap(self, tokens1, tokens2):
        """
        Compute keyword overlap percentage between two token lists
        
        Args:
            tokens1 (list): First token list
            tokens2 (list): Second token list
            
        Returns:
            float: Overlap percentage (0-1)
        """
        if not tokens1 or not tokens2:
            return 0.0
        
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_length_ratio(self, answer_length, reference_length):
        """
        Compute length ratio to check if answer is appropriately detailed
        
        Args:
            answer_length (int): User answer word count
            reference_length (int): Reference answer word count
            
        Returns:
            float: Length ratio score (0-1), 1 means similar length
        """
        if reference_length == 0:
            return 1.0
        
        ratio = answer_length / reference_length
        
        # Ideal is 0.7 to 1.3 (70%-130% of reference length)
        if 0.7 <= ratio <= 1.3:
            return 1.0
        elif 0.5 <= ratio < 0.7 or 1.3 < ratio <= 1.5:
            return 0.7
        elif 0.3 <= ratio < 0.5 or 1.5 < ratio <= 2.0:
            return 0.4
        else:
            return 0.2
    
    def evaluate_answer(self, question, user_answer, reference_answer):
        """
        Evaluate user answer using TF-IDF and cosine similarity with reference answer comparison
        
        Args:
            question (str): The interview question
            user_answer (str): User's answer
            reference_answer (dict): Reference answer dict with 'answer', 'human_score' (REQUIRED)
            
        Returns:
            dict: Evaluation results with score, similarity, and detailed feedback
        """
        # Preprocess
        question_tokens = self.preprocess_text(question)
        answer_tokens = self.preprocess_text(user_answer)
        
        # Basic validation
        if len(answer_tokens) == 0:
            return {
                'score': 0.0,
                'max_score': 10.0,
                'similarity': 0.0,
                'feedback': '❌ Empty or invalid answer',
                'details': {
                    'length_penalty': True,
                    'question_relevance': 0.0,
                    'reference_similarity': 0.0,
                    'keyword_overlap': 0.0,
                    'length_ratio': 0.0
                }
            }
        
        # Extract reference answer (REQUIRED)
        if isinstance(reference_answer, dict):
            ref_answer_text = reference_answer.get('answer', '')
            ref_human_score = reference_answer.get('human_score', 0)
        else:
            ref_answer_text = str(reference_answer)
            ref_human_score = 0
        
        reference_tokens = self.preprocess_text(ref_answer_text)
        
        # Prepare documents for IDF calculation (always 3 documents now)
        documents = [question_tokens, answer_tokens, reference_tokens]
        
        # Compute IDF for all documents
        idf_dict = self.compute_idf(documents)
        
        # Compute TF-IDF for question and answer
        question_tf = self.compute_tf(question_tokens)
        answer_tf = self.compute_tf(answer_tokens)
        
        question_tfidf = self.compute_tfidf(question_tf, idf_dict)
        answer_tfidf = self.compute_tfidf(answer_tf, idf_dict)
        
        # Calculate similarity with question (relevance check)
        question_relevance = self.cosine_similarity(question_tfidf, answer_tfidf)
        
        # Multi-way comparison with reference answer
        # 1. TF-IDF Cosine Similarity
        ref_tf = self.compute_tf(reference_tokens)
        ref_tfidf = self.compute_tfidf(ref_tf, idf_dict)
        reference_similarity = self.cosine_similarity(answer_tfidf, ref_tfidf)
        
        # 2. Keyword Overlap (Jaccard similarity)
        keyword_overlap = self.compute_keyword_overlap(answer_tokens, reference_tokens)
        
        # 3. Length Ratio
        answer_word_count = len(user_answer.split())
        ref_word_count = len(ref_answer_text.split())
        length_ratio = self.compute_length_ratio(answer_word_count, ref_word_count)
        
        # Scoring logic (0-10 scale) - improved for better base scores
        score = 0.0
        
        # 1. Length score (0-2.5 points) - more lenient
        word_count = len(user_answer.split())
        if word_count < 5:
            length_score = 0.5
            length_penalty = True
        elif word_count < 15:
            length_score = 1.5
            length_penalty = False
        elif word_count < 100:
            length_score = 2.5
            length_penalty = False
        else:
            length_score = 2.0
            length_penalty = False
        
        score += length_score
        
        # 2. Question relevance score (0-3.5 points) - improved baseline
        # Good answers should have some overlap with question terms
        relevance_score = min(3.5, question_relevance * 7.0 + 1.0)
        score += relevance_score
        
        # 3. Reference comparison score (0-4 points)
        # Combine multiple similarity metrics with improved weighting
        # TF-IDF similarity: 40%, Keyword overlap: 40%, Length ratio: 20%
        ref_score = (reference_similarity * 0.4 + keyword_overlap * 0.4 + length_ratio * 0.2) * 4.0 + 1.0
        score += min(4.0, ref_score)
        
        # Generate professional feedback
        if score >= 8.0:
            feedback = "Your answer demonstrates strong relevance and comprehensive understanding of the topic."
        elif score >= 6.0:
            feedback = "Your answer shows solid understanding. Consider adding more specific examples or technical details."
        elif score >= 4.0:
            feedback = "Your answer covers basic concepts but would benefit from greater technical depth and more detailed explanations."
        elif score >= 2.0:
            feedback = "Your answer needs more technical details and specific examples to demonstrate full understanding."
        else:
            feedback = "Your answer requires significant improvement in both content depth and technical accuracy."
        
        return {
            'score': round(score, 2),
            'max_score': 10.0,
            'similarity': round(question_relevance, 3),
            'feedback': feedback,
            'details': {
                'length_penalty': length_penalty,
                'length_score': round(length_score, 2),
                'question_relevance': round(relevance_score, 2),
                'reference_tfidf_similarity': round(reference_similarity, 3),
                'keyword_overlap': round(keyword_overlap, 3),
                'length_ratio': round(length_ratio, 3),
                'combined_ref_score': round(ref_score, 2),
                'word_count': word_count,
                'unique_terms': len(answer_tokens),
                'has_reference': bool(reference_tokens),
                'reference_human_score': ref_human_score
            }
        }


# Example usage
if __name__ == "__main__":
    evaluator = TFIDFAnswerEvaluator()
    
    # Example 1: Technical question
    question = "What is padding in CNNs and why is it used?"
    
    good_answer = """Padding in CNNs involves adding extra pixels around the border of an image 
    before applying convolution. It's used to preserve spatial dimensions and prevent information 
    loss at the edges of the image. Zero-padding is most common."""
    
    bad_answer = "I don't know"
    
    mediocre_answer = "Padding adds zeros to images in neural networks."
    
    print("="*70)
    print("EXAMPLE 1: Technical Question")
    print("="*70)
    print(f"Question: {question}\n")
    
    print("Good Answer:")
    result = evaluator.evaluate_answer(question, good_answer)
    print(f"Score: {result['score']}/10")
    print(f"Feedback: {result['feedback']}")
    print(f"Details: {result['details']}\n")
    
    print("Bad Answer:")
    result = evaluator.evaluate_answer(question, bad_answer)
    print(f"Score: {result['score']}/10")
    print(f"Feedback: {result['feedback']}")
    print(f"Details: {result['details']}\n")
    
    print("Mediocre Answer:")
    result = evaluator.evaluate_answer(question, mediocre_answer)
    print(f"Score: {result['score']}/10")
    print(f"Feedback: {result['feedback']}")
    print(f"Details: {result['details']}\n")
