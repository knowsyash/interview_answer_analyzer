from dataset_loader import DatasetLoader
from evaluator import AnswerEvaluator
from tfidf_evaluator import TFIDFAnswerEvaluator
from reference_answer_loader import ReferenceAnswerLoader
from resources import get_tip
from logger import log_response
from evaluate_model import InterviewBotEvaluator
import numpy as np
import random
import pandas as pd
import os

def get_interview_category():
    """Select interview category - first choose technical vs non-technical"""
    print("\n" + "="*70)
    print("STEP 1: SELECT INTERVIEW TYPE")
    print("="*70)
    print("\n1. Technical Domain")
    print("   � Technology-focused interview questions")
    print("   � Web Development, Programming, etc.")
    
    print("\n2. Non-Technical Domain")
    print("   👥 Behavioral & Soft Skills questions")
    print("   🎯 STAR format, Leadership, Communication, etc.")
    print("="*70)

    # First ask technical vs non-technical
    while True:
        type_choice = input("\nAre you interested in Technical or Non-Technical domain? (1-2): ").strip()
        if type_choice in ["1", "2"]:
            break
        else:
            print("❌ Invalid choice. Please choose 1 or 2.")
    
    # Then ask specific domain
    print("\n" + "="*70)
    print("STEP 2: SELECT YOUR DOMAIN")
    print("="*70)
    
    if type_choice == "1":
        # Technical domains
        print("\nAvailable Technical Domains:")
        print("\n1. Web Development")
        print("   📊 44 Q&A pairs covering HTML, CSS, JavaScript, React, Node.js")
        print("   📚 Expert answers with human scores (8-10/10)")
        print("   ✅ Best for: Frontend & Backend interview preparation")
        print("   🎯 Each answer compared with expert reference answers")
        print("="*70)
        
        while True:
            domain_choice = input("\nWhich technical domain are you interested in? (1): ").strip()
            if domain_choice == "1":
                return "webdev"
            else:
                print("❌ Invalid choice. Please choose 1.")
    
    else:
        # Non-technical domains
        print("\nAvailable Non-Technical Domains:")
        print("\n1. Behavioral Questions (STAR Format)")
        print("   📊 9 unique behavioral questions across different roles")
        print("   📚 1,470 expert answer examples with human scores")
        print("   ✅ Best for: Learning STAR format, reference comparison")
        print("   🎯 Each answer compared against 100+ reference examples")
        print("="*70)
        
        while True:
            domain_choice = input("\nWhich non-technical domain are you interested in? (1): ").strip()
            if domain_choice == "1":
                return "behavioral"
            else:
                print("❌ Invalid choice. Please choose 1.")

def get_technical_subcategory():
    """This function is no longer used - removed technical question-only datasets"""
    return None

def load_technical_questions(role_dataset):
    """Load technical questions from the selected role dataset"""
    if not role_dataset:
        return []
    
    csv_path = role_dataset['file']
    
    if not os.path.exists(csv_path):
        print(f"❌ Dataset file not found: {csv_path}")
        return []
    
    df = pd.read_csv(csv_path)
    questions = []
    
    # Check if dataset has reference answers (like behavioral)
    has_answers = 'answer' in df.columns and 'competency' in df.columns
    
    # Handle different CSV formats
    if 'DESCRIPTION' in df.columns:
        # Deep Learning questions format (questions only)
        for _, row in df.iterrows():
            questions.append({
                'id': row.get('ID', len(questions) + 1),
                'question': row['DESCRIPTION'],
                'type': 'technical',
                'role': role_dataset['name'],
                'has_reference': False
            })
    elif 'question' in df.columns:
        # General question format
        for _, row in df.iterrows():
            q_data = {
                'id': len(questions) + 1,
                'question': row['question'],
                'type': 'technical',
                'role': role_dataset['name'],
                'has_reference': has_answers
            }
            
            # If has answers, include them for reference comparison
            if has_answers:
                q_data['answer'] = row.get('answer', '')
                q_data['competency'] = row.get('competency', '')
                q_data['human_score'] = row.get('human_score', 0)
            
            questions.append(q_data)
    
    return questions

def handle_technical_answer(question, tfidf_evaluator, all_scores, ref_loader=None):
    """Handle answer for technical questions with TF-IDF based scoring"""
    user_answer = input("\n💬 Your Answer: ").strip()
    
    if not user_answer:
        print("⚠️ Skipped.")
        return False
    
    print("\n" + "="*70)
    print("📊 EVALUATING YOUR ANSWER (TF-IDF Analysis)")
    print("="*70)
    
    # Check if question has its own reference answer (like web dev Q&A dataset)
    reference_answer = None
    if question.get('has_reference'):
        # Use question's own reference answer
        reference_answer = {
            'answer': question['answer'],
            'competency': question['competency'],
            'human_score': question.get('human_score', 0)
        }
        print(f"📚 Using reference answer from dataset (Human Score: {question.get('human_score', 'N/A')}/10)")
    elif ref_loader:
        # Try to get reference from behavioral dataset
        reference_answer = ref_loader.get_reference_answer(
            question['question'],
            competency="Technical Skills"
        )
    
    # Use TF-IDF evaluator to score the answer
    question_text = question['question']
    result = tfidf_evaluator.evaluate_answer(question_text, user_answer, reference_answer)
    
    # Extract results
    total_score = result['score']
    feedback = result['feedback']
    details = result['details']
    
    # Display detailed breakdown
    print(f"\n📝 Answer Statistics:")
    print(f"   • Word count: {details['word_count']} words")
    print(f"   • Unique terms: {details['unique_terms']} terms")
    print(f"   • Length penalty: {'Yes ⚠️' if details['length_penalty'] else 'No ✅'}")
    
    print(f"\n🔍 TF-IDF Score Breakdown:")
    print(f"   • Length Score: {details['length_score']}/2.0")
    print(f"   • Question Relevance: {details['question_relevance']}/3.0")
    
    if details['has_reference']:
        print(f"   • Reference Comparison: {details['combined_ref_score']}/5.0")
        print(f"\n📊 Reference Answer Comparison:")
        print(f"   • TF-IDF Similarity: {details['reference_tfidf_similarity']:.3f}")
        print(f"   • Keyword Overlap: {details['keyword_overlap']:.3f}")
        print(f"   • Length Ratio: {details['length_ratio']:.3f}")
        if details['reference_human_score']:
            print(f"   • Reference Human Score: {details['reference_human_score']}/10")
    else:
        print(f"   • Technical Depth: {details['combined_ref_score']}/5.0")
    
    print(f"   " + "-" * 50)
    print(f"   TOTAL SCORE        : {total_score:.2f}/10.0")
    
    # Display feedback
    print(f"\n{feedback}")
    
    print("\n💡 TIPS FOR BETTER ANSWERS:")
    print("   • Use technical terminology relevant to the concept")
    print("   • Explain HOW and WHY, not just WHAT")
    print("   • Include practical examples or use cases")
    print("   • Mention advantages/disadvantages if applicable")
    print("="*70)
    
    all_scores.append(total_score)
    log_response(question['question'], user_answer, total_score, feedback)
    
    return True

def handle_behavioral_answer(question, role, evaluator, all_scores, all_human_comparisons, ref_loader=None):
    """Handle answer for behavioral questions with reference answer comparison"""
    user_answer = input("\n💬 Your Answer: ").strip()
    
    if not user_answer:
        print("⚠️ Skipped.")
        return False
    
    # Try to get reference answer from loader
    reference_answer = None
    if ref_loader:
        # Extract competency from question if available
        competency = None
        if 'competency' in question:
            comp_str = question['competency']
            if isinstance(comp_str, str) and '[' in comp_str:
                # Parse list string
                competencies = [c.strip().strip("'\"[]") for c in comp_str.split(',')]
                competency = competencies[0] if competencies else None
            else:
                competency = str(comp_str)
        
        reference_answer = ref_loader.get_reference_answer(
            question['question'],
            competency=competency
        )
    
    # Get the expected answer (if available)
    expected_answer = question.get("answer", "")
    
    # Get comprehensive evaluation
    score, feedback = evaluator.evaluate_answer(
        user_answer, 
        expected_answer,
        question.get("keywords", [])
    )
    
    # Compare with human evaluations (only if answer is available)
    human_comparison = None
    if expected_answer:
        human_comparison = evaluator.compare_with_human_evaluation(
            role, 
            question["question"], 
            expected_answer
        )
    
    # Print detailed feedback
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    print(f"Overall Score      : {feedback['final_score']:.2f}/10")
    print(f"Semantic Score     : {feedback['semantic_score']:.2f}/10")
    print(f"Structure Score    : {feedback['structure_score']:.2f}/10")
    print(f"Keyword Coverage   : {feedback['keyword_score']:.2f}/10")
    
    # Show reference answer comparison if available
    if reference_answer:
        print(f"\n🎯 Reference Answer Comparison:")
        print(f"   • Reference Human Score: {reference_answer.get('human_score', 'N/A')}/10")
        print(f"   • Reference Competencies: {', '.join(reference_answer.get('competency', []))}")
    
    # Show human score comparison
    if 'human_score' in question:
        print(f"\nHuman Expert Score : {question['human_score']}/4 (Rating scale: 1-4)")
        ai_score_scaled = feedback['final_score'] / 2.5  # Convert 10-point to 4-point scale
        print(f"AI Score (scaled)  : {ai_score_scaled:.2f}/4")
        print(f"Difference         : {abs(ai_score_scaled - question['human_score']):.2f}")
    
    if feedback['improvements']:
        print("\n💡 SUGGESTIONS FOR IMPROVEMENT:")
        for imp in feedback['improvements']:
            print(f"   • {imp}")
    
    # Add technical terms suggestion
    print("\n📝 TIP: Include technical terms like:")
    competencies = question.get('competency', '')
    if 'Technical' in competencies or 'Analysis' in competencies:
        print("   • Specific methodologies, tools, or frameworks you used")
        print("   • Quantitative metrics and data-driven insights")
        print("   • Technical challenges and innovative solutions")
    if 'Communication' in competencies or 'Negotiation' in competencies:
        print("   • Stakeholder engagement strategies")
        print("   • Clear communication of complex concepts")
        print("   • Conflict resolution techniques")
    if 'Leadership' in competencies or 'Management' in competencies:
        print("   • Team coordination and delegation approaches")
        print("   • Strategic planning and execution")
        print("   • Performance improvement initiatives")
    print("="*70)
    
    all_scores.append(feedback['final_score'])
    if human_comparison:
        all_human_comparisons.append(human_comparison)
    
    # Create feedback summary for logging
    feedback_summary = f"Overall: {feedback['final_score']:.2f}/10, Semantic: {feedback['semantic_score']:.2f}/10, Structure: {feedback['structure_score']:.2f}/10"
    log_response(question['question'], user_answer, feedback['final_score'], feedback_summary)
    
    return True

def show_session_summary(questions, attempted, all_scores, all_human_comparisons):
    print("\n===== SESSION SUMMARY =====")
    print(f"Total Questions    : {len(questions)}")
    print(f"Attempted          : {attempted}")
    
    if all_scores:
        avg_score = np.mean(all_scores)
        print(f"\nPerformance Metrics:")
        print(f"Average Score      : {avg_score:.2f}")
        
        if avg_score >= 8:
            performance = "✅ Excellent"
        elif avg_score >= 6:
            performance = "✅ Good"
        elif avg_score >= 4:
            performance = "⚠️ Fair"
        else:
            performance = "❌ Needs Improvement"
        
        print(f"Overall Rating     : {performance}")

        # Show human evaluation comparison if available
        if all_human_comparisons:
            correlations = [comp['correlation'] for comp in all_human_comparisons if not np.isnan(comp['correlation'])]
            maes = [comp['mean_absolute_error'] for comp in all_human_comparisons]
            
            if correlations:
                avg_correlation = np.mean(correlations)
                avg_mae = np.mean(maes)
                
                print("\nHuman Evaluation Comparison:")
                print(f"Correlation       : {avg_correlation:.2f}")
                print(f"Mean Abs. Error   : {avg_mae:.2f}")
                
                if avg_correlation > 0.7:
                    print("✅ High agreement with human evaluators")
                elif avg_correlation > 0.5:
                    print("⚠️ Moderate agreement with human evaluators")
                else:
                    print("❌ Low agreement with human evaluators")

        # Run model evaluation for comparison (only for behavioral)
        if all_human_comparisons:
            print("\n===== MODEL EVALUATION =====")
            model_evaluator = InterviewBotEvaluator()
            evaluation_results = model_evaluator.evaluate_model()
            
            print("\nModel Performance Metrics:")
            print(evaluation_results)
            print("\nDetailed evaluation results and visualizations have been saved to the model_evaluation directory.")
            
            # Compare human vs model performance
            print("\n===== HUMAN VS MODEL COMPARISON =====")
            model_accuracy = evaluation_results['Accuracy'].mean()
            print(f"Your Score        : {round(avg_score * 10, 2)}/10")
            print(f"Model Accuracy    : {round(model_accuracy, 2)}%")
            print(f"Coverage          : {round(evaluation_results['Coverage'].mean(), 2)}%")
            
            if avg_score * 10 > model_accuracy:
                print("🎉 Great job! You performed better than the model average!")
            elif avg_score * 10 == model_accuracy:
                print("📊 Your performance matches the model's accuracy!")
            else:
                print("💪 Keep practicing! The model suggests room for improvement.")
    else:
        print("No questions were attempted.")
    print("=============================")

def run_interview_session():
    print("="*70)
    print(" AI-POWERED INTERVIEW COACH BOT")
    print("="*70)
    
    # Initialize evaluators
    behavioral_evaluator = AnswerEvaluator()
    tfidf_evaluator = TFIDFAnswerEvaluator()
    
    # Initialize reference answer loader
    print("\n🔄 Loading reference answer database...")
    ref_loader = ReferenceAnswerLoader()
    ref_loaded = ref_loader.load_reference_answers()
    
    if ref_loaded:
        print("✅ Reference answers loaded successfully!")
        print(f"📊 {len(ref_loader.competency_answers)} competency categories available")
    else:
        print("⚠️  No reference answers available - using standard evaluation")
        ref_loader = None
    
    # Get category
    category = get_interview_category()
    
    # Load Questions using smart dataset loader
    print("\n🔍 Loading questions...")
    loader = DatasetLoader()
    
    if category == "behavioral":
        # Start behavioral interview
        print("\n" + "="*70)
        print("BEHAVIORAL INTERVIEW - STAR FORMAT")
        print("="*70)
        print("📊 9 unique behavioral questions from different roles")
        print("📚 1,470 expert answer examples with human scores")
        print("🎯 You'll answer 3 questions, compared against references")
        print("="*70)
        
        all_questions_data = loader.get_interview_questions_dataset(format='json')
    elif category == "webdev":
        # Start web development interview
        print("\n" + "="*70)
        print("WEB DEVELOPMENT INTERVIEW")
        print("="*70)
        print("� 44 Q&A pairs covering modern web technologies")
        print("📚 Expert answers scored 8-10/10")
        print("🎯 You'll answer 3 questions, compared against expert answers")
        print("="*70)
        
        # Load webdev_interview_qa.csv
        webdev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'webdev_interview_qa.csv')
        if os.path.exists(webdev_path):
            df = pd.read_csv(webdev_path)
            all_questions_data = []
            for _, row in df.iterrows():
                all_questions_data.append({
                    'question': row['question'],
                    'answer': row['answer'],
                    'competency': row['competency'],
                    'human_score': row['human_score']
                })
        else:
            print("❌ Web development dataset not found!")
            return
    else:
        print("❌ Invalid category!")
        return
    
    if not all_questions_data:
        print("❌ No interview questions dataset found!")
        return
    
    # Get all unique questions
    unique_questions_dict = {}
    for q in all_questions_data:
        question_text = q['question']
        if question_text not in unique_questions_dict:
            unique_questions_dict[question_text] = q
    
    unique_questions = list(unique_questions_dict.values())
    print(f"✅ Loaded {len(unique_questions)} unique questions")
    
    # Select 3 random unique questions
    NUM_QUESTIONS = 3
    if len(unique_questions) < NUM_QUESTIONS:
        NUM_QUESTIONS = len(unique_questions)
    
    selected_questions = random.sample(unique_questions, NUM_QUESTIONS)
    
    print("\n" + "="*70)
    print(f"{category.upper()} INTERVIEW SESSION")
    print("="*70)
    print(f"Questions: {NUM_QUESTIONS} unique questions")
    print(f"Reference Answers: {len(all_questions_data)} total examples")
    print("="*70)
    
    all_scores = []
    all_human_comparisons = []
    attempted = 0
    
    # Ask each unique question
    for i, question in enumerate(selected_questions, 1):
        print(f"\n{'='*70}")
        print(f"Q{i}/{NUM_QUESTIONS}: {question['question']}")
        print(f"{'='*70}")
        
        # Extract role from question for compatibility
        role = "General"
        if "role as a" in question['question']:
            role_part = question['question'].split("role as a ")[-1]
            role = role_part.strip()
        
        if handle_behavioral_answer(question, role, behavioral_evaluator, all_scores, all_human_comparisons, ref_loader):
            attempted += 1

    # Show final summary
    show_session_summary(selected_questions, attempted, all_scores, all_human_comparisons)

if __name__ == "__main__":
    run_interview_session()
