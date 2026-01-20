from flask import Flask, render_template, request, jsonify
import random
import pandas as pd
import os
import nltk
from random_forest_evaluator import RandomForestAnswerEvaluator
from reference_answer_loader import ReferenceAnswerLoader
from tfidf_evaluator import TFIDFAnswerEvaluator

# Download NLTK data at startup
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(resource, quiet=True)
    except:
        pass

app = Flask(__name__)

class InterviewBotSession:
    def __init__(self, category='webdev'):
        self.category = category
        self.current_question_idx = 0
        self.questions = []
        self.scores = []
        self.answers = []
        self.current_question = None
        self.current_answer = None
        self.rf_evaluator = None
        self.ref_loader = None
        self.tfidf_evaluator = None
        
        # Initialize evaluators
        self._initialize_evaluators()
        
        # Load questions
        self._load_questions()
    
    def _initialize_evaluators(self):
        """Initialize the evaluators"""
        try:
            self.rf_evaluator = RandomForestAnswerEvaluator()
            # Load the trained Random Forest model
            self.rf_evaluator.load_model()
            self.ref_loader = ReferenceAnswerLoader()
            self.tfidf_evaluator = TFIDFAnswerEvaluator()
        except Exception as e:
            print(f"Error initializing evaluators: {e}")
            # Set rf_evaluator to None if model loading fails
            self.rf_evaluator = None
    
    def _load_questions(self):
        """Load questions based on category"""
        # Get dataset based on category
        dataset_map = {
            'webdev': 'webdev_interview_qa.csv',
            'python': 'stackoverflow_training_data.csv',
            'java': 'stackoverflow_training_data.csv',
            'csharp': 'stackoverflow_training_data.csv',
            'javascript': 'stackoverflow_training_data.csv',
            'database': 'stackoverflow_training_data.csv',
            'behavioral': 'interview_data_with_scores.csv'  # Using proper behavioral dataset with competency data
        }
        
        csv_file = dataset_map.get(self.category, 'webdev_interview_qa.csv')
        
        # Try multiple possible paths for flexibility (local and deployed)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(base_dir, 'real_dataset_score', csv_file),
            os.path.join(base_dir, '..', 'real_dataset_score', csv_file),
            os.path.join('real_dataset_score', csv_file)
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Filter by technology if Stack Overflow data
            if self.category in ['python', 'java', 'csharp', 'javascript', 'database']:
                if 'tags' in df.columns:
                    # Filter by technology tag in stackoverflow data
                    tech_map = {
                        'python': 'python',
                        'java': 'java',
                        'csharp': 'c#',
                        'javascript': 'javascript',
                        'database': 'sql'
                    }
                    tech_tag = tech_map.get(self.category, self.category)
                    df = df[df['tags'].str.contains(tech_tag, case=False, na=False)]
            
            # Load questions
            for _, row in df.iterrows():
                # Handle different column names across datasets
                answer_col = None
                if 'answer' in df.columns:
                    answer_col = 'answer'
                elif 'user_answer' in df.columns:
                    answer_col = 'user_answer'
                
                # Get answer value safely
                answer_value = ''
                if answer_col:
                    ans = row.get(answer_col)
                    if pd.notna(ans):
                        answer_value = str(ans)
                
                # Get human score from either column name
                human_score = 0
                if 'human_score' in df.columns and pd.notna(row.get('human_score')):
                    human_score = float(row.get('human_score'))
                elif 'score' in df.columns and pd.notna(row.get('score')):
                    human_score = float(row.get('score'))
                
                q_data = {
                    'question': row['question'],
                    'answer': answer_value,
                    'competency': row.get('competency', ''),
                    'human_score': human_score,
                    'has_reference': bool(answer_value)
                }
                self.questions.append(q_data)
            
            # Shuffle and select subset
            random.shuffle(self.questions)
            num_questions = min(5, len(self.questions))
            self.questions = self.questions[:num_questions]
    
    def get_next_question(self):
        """Get the next question"""
        if self.current_question_idx < len(self.questions):
            self.current_question = self.questions[self.current_question_idx]
            return self.current_question['question']
        return None
    
    def evaluate_answer(self, answer):
        """Evaluate the user's answer using both TF-IDF and Random Forest"""
        if not self.current_question:
            return None
        
        self.current_answer = answer
        
        # Get reference answer if available
        reference_answer = None
        if self.current_question.get('has_reference'):
            reference_answer = {
                'answer': self.current_question['answer'],
                'competency': self.current_question['competency'],
                'human_score': self.current_question.get('human_score', 0)
            }
        
        # Evaluate with both models and combine scores
        tfidf_result = None
        rf_result = None
        
        # Get TF-IDF score
        tfidf_result = self.tfidf_evaluator.evaluate_answer(
            self.current_question['question'],
            answer,
            reference_answer
        )
        tfidf_score = tfidf_result['score']
        tfidf_feedback = tfidf_result['feedback']
        
        # Get Random Forest score if model is available
        if self.rf_evaluator and self.rf_evaluator.model is not None:
            rf_result = self.rf_evaluator.evaluate_answer(
                self.current_question['question'],
                answer,
                reference_answer
            )
            rf_score = rf_result['predicted_score']
            rf_feedback = rf_result['feedback']
            
            # Combine scores (weighted average: 50% TF-IDF + 50% Random Forest)
            final_score = (tfidf_score * 0.5) + (rf_score * 0.5)
            
            # Build professional combined feedback
            feedback_parts = [
                f"Overall Score: {final_score:.2f}/10.0",
                f"\n\nTF-IDF Analysis Score: {tfidf_score:.2f}/10.0",
                f"\n{tfidf_feedback}",
                f"\n\nMachine Learning Model Score: {rf_score:.2f}/10.0",
                f"\n{rf_feedback}"
            ]
            
            feedback = ''.join(feedback_parts)
        else:
            # Fallback to TF-IDF only if Random Forest not available
            final_score = tfidf_score
            feedback = f"Score: {tfidf_score:.2f}/10.0\n\n{tfidf_feedback}"
        
        # Store score
        self.scores.append(final_score)
        self.answers.append({
            'question': self.current_question['question'],
            'answer': answer,
            'score': final_score,
            'feedback': feedback,
            'tfidf_score': tfidf_score,
            'rf_score': rf_score if rf_result else None
        })
        
        return {
            'score': final_score,
            'feedback': feedback,
            'reference_answer': self.current_question.get('answer', 'No reference answer available.')
        }
    
    def move_to_next(self):
        """Move to the next question"""
        self.current_question_idx += 1
        return self.current_question_idx < len(self.questions)
    
    def get_summary(self):
        """Get session summary"""
        if not self.scores:
            return {
                'total_questions': len(self.questions),
                'attempted': 0,
                'average_score': 0,
                'message': 'No questions attempted yet.'
            }
        
        avg_score = sum(self.scores) / len(self.scores)
        
        return {
            'category': self.category.upper(),
            'total_questions': len(self.questions),
            'attempted': len(self.scores),
            'average_score': f"{avg_score:.2f}",
            'highest_score': f"{max(self.scores):.2f}",
            'lowest_score': f"{min(self.scores):.2f}",
            'performance': 'Excellent' if avg_score >= 8 else 'Good' if avg_score >= 6 else 'Needs Improvement'
        }

# Global session storage
sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize():
    try:
        data = request.json
        category = data.get('category', 'webdev')
        session_id = str(random.randint(1000, 9999))
        
        # Create new session
        sessions[session_id] = InterviewBotSession(category)
        
        return jsonify({
            'status': 'success',
            'message': f'Interview Bot initialized for {category.upper()} domain!',
            'session_id': session_id,
            'total_questions': len(sessions[session_id].questions)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in sessions:
            return jsonify({'status': 'error', 'message': 'Invalid session. Please initialize first.'})
        
        session = sessions[session_id]
        question = session.get_next_question()
        
        if question:
            return jsonify({
                'status': 'success',
                'question': question,
                'question_number': session.current_question_idx + 1,
                'total_questions': len(session.questions)
            })
        else:
            return jsonify({
                'status': 'complete',
                'message': 'No more questions available.'
            })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/submit', methods=['POST'])
def submit_answer():
    try:
        data = request.json
        session_id = data.get('session_id')
        answer = data.get('answer', '')
        
        if session_id not in sessions:
            return jsonify({'status': 'error', 'message': 'Invalid session.'})
        
        session = sessions[session_id]
        result = session.evaluate_answer(answer)
        
        if result:
            return jsonify({
                'status': 'success',
                'score': result['score'],
                'feedback': result['feedback'],
                'reference_answer': result['reference_answer']
            })
        else:
            return jsonify({'status': 'error', 'message': 'No current question to evaluate.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/next', methods=['POST'])
def next_question():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in sessions:
            return jsonify({'status': 'error', 'message': 'Invalid session.'})
        
        session = sessions[session_id]
        has_more = session.move_to_next()
        
        if has_more:
            return jsonify({
                'status': 'success',
                'message': 'Ready for next question'
            })
        else:
            return jsonify({
                'status': 'complete',
                'message': 'Interview session completed!'
            })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/summary', methods=['POST'])
def get_summary():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in sessions:
            return jsonify({'status': 'error', 'message': 'Invalid session.'})
        
        session = sessions[session_id]
        summary = session.get_summary()
        
        return jsonify({
            'status': 'success',
            'summary': summary
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
