from flask import Flask, render_template, request, jsonify
import random
import pandas as pd
import os
from dataset_loader import DatasetLoader
from random_forest_evaluator import RandomForestAnswerEvaluator
from reference_answer_loader import ReferenceAnswerLoader
from tfidf_evaluator import TFIDFAnswerEvaluator

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
            self.ref_loader = ReferenceAnswerLoader()
            self.tfidf_evaluator = TFIDFAnswerEvaluator()
        except Exception as e:
            print(f"Error initializing evaluators: {e}")
    
    def _load_questions(self):
        """Load questions based on category"""
        loader = DatasetLoader()
        
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
        csv_path = os.path.join('data', 'real_dataset_score', csv_file)
        
        if os.path.exists(csv_path):
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
        """Evaluate the user's answer"""
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
        
        # Use TF-IDF evaluator
        result = self.tfidf_evaluator.evaluate_answer(
            self.current_question['question'],
            answer,
            reference_answer
        )
        
        score = result['score']
        feedback = result['feedback']
        
        # Store score
        self.scores.append(score)
        self.answers.append({
            'question': self.current_question['question'],
            'answer': answer,
            'score': score,
            'feedback': feedback
        })
        
        return {
            'score': score,
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
    app.run(debug=True, port=5000)
