from dataset_loader import DatasetLoader
from evaluator import AnswerEvaluator
from random_forest_evaluator import RandomForestAnswerEvaluator
from reference_answer_loader import ReferenceAnswerLoader
from resources import get_tip
from logger import log_response
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
    valid_type_inputs = {
        '1': 'technical', 'technical': 'technical', 'tech': 'technical',
        '2': 'non-technical', 'non-technical': 'non-technical', 'nontechnical': 'non-technical', 'non': 'non-technical', 'behavioral': 'non-technical'
    }

    while True:
        type_choice = input("\nAre you interested in Technical or Non-Technical domain? (1-2 or 'technical'/'behavioral'): ").strip().lower()
        mapped = valid_type_inputs.get(type_choice)
        if mapped:
            type_choice = '1' if mapped == 'technical' else '2'
            break
        else:
            print("❌ Invalid choice. Please choose 1, 2, or type 'technical'/'behavioral'.")
    
    # Then ask specific domain
    print("\n" + "="*70)
    print("STEP 2: SELECT YOUR DOMAIN")
    print("="*70)
    
    if type_choice == "1":
        # Technical domains - Multiple options from Stack Overflow + Web Dev
        print("\n🎯 TECHNICAL DOMAINS - WITH REFERENCE ANSWERS & SCORES:")
        print("="*70)
        
        print("\n1. Web Development (Frontend/Backend)")
        print("   📊 44 expert Q&A pairs - HTML, CSS, JavaScript, React, Node.js")
        print("   ✅ Human scores: 8-10/10")
        
        print("\n2. Python Programming")
        print("   📊 1,288 Stack Overflow Q&A - Django, Flask, Pandas, NumPy")
        print("   ✅ Community-validated answers with scores")
        
        print("\n3. Java Development")
        print("   📊 2,711 Stack Overflow Q&A - Spring, Hibernate, Maven, Android")
        print("   ✅ Enterprise & Android development questions")
        
        print("\n4. C# / .NET Development")
        print("   📊 3,022 Stack Overflow Q&A - ASP.NET, Entity Framework, WPF")
        print("   ✅ Microsoft stack & Windows development")
        
        print("\n5. JavaScript / Node.js")
        print("   📊 1,026 Stack Overflow Q&A - React, Angular, TypeScript, Node")
        print("   ✅ Modern web frameworks & full-stack JS")
        
        print("\n6. Database / SQL")
        print("   📊 1,025 Stack Overflow Q&A - MySQL, PostgreSQL, MongoDB, Oracle")
        print("   ✅ SQL queries, database design, optimization")
        
        print("="*70)
        
        while True:
            domain_choice = input("\nSelect domain (1-6 or type name like 'python'/'java' or 'back'): ").strip().lower()
            
            domain_map = {
                '1': 'webdev', 'webdev': 'webdev', 'web': 'webdev', 'web development': 'webdev',
                '2': 'python', 'python': 'python', 'py': 'python',
                '3': 'java', 'java': 'java',
                '4': 'csharp', 'c#': 'csharp', 'csharp': 'csharp', '.net': 'csharp', 'dotnet': 'csharp',
                '5': 'javascript', 'javascript': 'javascript', 'js': 'javascript', 'node': 'javascript', 'nodejs': 'javascript',
                '6': 'database', 'database': 'database', 'sql': 'database', 'db': 'database'
            }
            
            if domain_choice in domain_map:
                return domain_map[domain_choice]
            elif domain_choice == 'back':
                return get_interview_category()
            else:
                print("❌ Invalid choice. Please choose 1-6, type domain name, or 'back'.")
    
    else:
        # Non-technical domains
        print("\n🎯 NON-TECHNICAL DOMAINS - WITH REFERENCE ANSWERS:")
        print("="*70)
        print("\n1. Behavioral Questions (STAR Format)")
        print("   📊 1,470 Q&A pairs across 21 competency categories")
        print("   ✅ Has reference answers - Similarity checking enabled")
        print("   📚 Expert STAR format examples with human scores (1-5)")
        print("   🎯 Best for: Learning STAR format, behavioral interviews")
        print("   💡 Each answer compared against 100+ reference examples")
        print("="*70)
        
        while True:
            domain_choice = input("\nWhich non-technical domain are you interested in? (1 or type 'behavioral'): ").strip().lower()
            if domain_choice in ("1", "behavioral", "behavior"):
                return "behavioral"
            else:
                print("❌ Invalid choice. Please choose 1 or type 'behavioral'.")

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

def handle_behavioral_answer(question, role, evaluator, all_scores, all_human_comparisons, ref_loader=None, domain="behavioral"):
    """Handle answer for behavioral questions with reference answer comparison"""
    user_answer = input("\n💬 Your Answer: ").strip()
    
    if not user_answer:
        print("⚠️ Skipped.")
        return False
    
    # Determine if this is a technical domain (shorter answers acceptable)
    is_technical = domain in ["webdev", "ml_ai", "software_engineering", "deep_learning"]
    
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
    
    # Check if using Random Forest evaluator
    if hasattr(evaluator, 'model') and evaluator.model is not None:
        # Check answer quality first
        word_count = len(user_answer.split())
        
        # Domain-aware thresholds
        if is_technical:
            min_words = 2  # "404" or "CSS" can be valid
            brief_threshold = 5
            short_threshold = 10
        else:
            min_words = 3
            brief_threshold = 10
            short_threshold = 20
        
        # Strict validation for extremely poor answers
        if word_count < min_words:
            # Single word answers (or 2 words for behavioral)
            score = 1
            confidence = 1.0
            feedback_list = [
                f"❌ Answer is too short ({word_count} word{'s' if word_count > 1 else ''}) - This is unacceptable",
                "Provide a complete explanation with proper sentences" if not is_technical else "Provide at least the key term or concept",
                f"Minimum expected: {10 if is_technical else 20}-30 words",
                "Include: definition, explanation, examples, or comparisons"
            ]
            score_10 = 1.0
            
        elif word_count < brief_threshold:
            # Very brief answer - check similarity for technical domains
            if is_technical and expected_answer:
                # For technical questions, short answers might be correct
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                
                try:
                    vectorizer = TfidfVectorizer(stop_words='english')
                    vectors = vectorizer.fit_transform([user_answer.lower(), expected_answer.lower()])
                    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                    
                    if similarity > 0.6:
                        # High similarity - correct short answer!
                        score = 4
                        confidence = 0.9
                        score_10 = 8.0
                        feedback_list = [
                            f"✓ Correct! ({word_count} words)",
                            "💡 Consider adding an example or elaboration to demonstrate deeper understanding"
                        ]
                    elif similarity > 0.3:
                        # Partially correct
                        score = 3
                        confidence = 0.7
                        score_10 = 6.0
                        feedback_list = [
                            f"⚠️ Partially correct ({word_count} words)",
                            "Add more detail or clarify your answer",
                            "Include the complete concept or additional context"
                        ]
                    else:
                        # Low similarity - likely wrong
                        score = 1
                        confidence = 0.9
                        score_10 = 2.5
                        feedback_list = [
                            f"❌ Answer appears incorrect ({word_count} words)",
                            "Review the question and provide accurate information",
                            "Add proper explanation (10-20 words)"
                        ]
                except:
                    score = 2
                    confidence = 0.6
                    score_10 = 4.0
                    feedback_list = [
                        f"⚠️ Answer is brief ({word_count} words)",
                        "Add 5-10 more words with explanation"
                    ]
            else:
                # Behavioral or no reference - insufficient detail
                score = 1
                confidence = 0.9
                feedback_list = [
                    f"❌ Answer is too brief ({word_count} words) - Insufficient detail",
                    f"Expand your explanation significantly (aim for {25 if is_technical else 40}-50 words minimum)",
                    "Add: what it is, how it works, why it matters, examples",
                    "Current answer lacks depth and completeness"
                ]
                score_10 = 2.5
            
        elif word_count < short_threshold:
            # Short answer - use RF with domain-appropriate penalty
            rf_result = evaluator.evaluate_answer(
                user_answer,
                question['question'],
                return_details=True
            )
            base_score = rf_result['score']
            
            if is_technical:
                # Technical: lighter penalty, might be sufficient
                score = max(2, min(4, base_score - 0.5))
                penalty_msg = f"Good answer ({word_count} words). Could add examples for excellence."
            else:
                # Behavioral: heavy penalty, needs STAR format
                score = max(1, min(2, base_score - 2))
                penalty_msg = f"⚠️ Answer is too short ({word_count} words). Add 20-30 more words with examples and details."
            
            confidence = rf_result['confidence'] * 0.8
            features = rf_result['features']
            score_10 = score * 2.0
            feedback_list = evaluator.get_feedback(score, features)
            feedback_list.insert(0, penalty_msg)
            
        elif word_count < 35:
            # Acceptable length but could be better (20-34 words)
            rf_result = evaluator.evaluate_answer(
                user_answer,
                question['question'],
                return_details=True
            )
            base_score = rf_result['score']
            
            # Check similarity with reference answer if available
            if expected_answer and len(expected_answer) > 20:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                try:
                    vectorizer = TfidfVectorizer(stop_words='english')
                    vectors = vectorizer.fit_transform([user_answer.lower(), expected_answer.lower()])
                    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                    
                    # Boost score if good similarity
                    if similarity > 0.3:
                        similarity_boost = min(1.0, (similarity - 0.3) * 2)  # 0 to 1 boost
                        score = min(4, base_score + similarity_boost)
                    else:
                        score = max(2, min(3, base_score - 0.5))
                except:
                    score = max(2, min(3, base_score - 0.5))
            else:
                # Moderate penalty for brevity
                score = max(2, min(3, base_score - 0.5))
            
            confidence = rf_result['confidence']
            features = rf_result['features']
            score_10 = score * 2.0
            feedback_list = evaluator.get_feedback(score, features)
            if score < 4:
                feedback_list.insert(0, f"Answer is acceptable ({word_count} words) but could be more comprehensive. Add examples or technical details.")
            
        else:
            # Good length (35+ words) - use full RF evaluation with reference check
            rf_result = evaluator.evaluate_answer(
                user_answer,
                question['question'],
                return_details=True
            )
            base_score = rf_result['score']
            
            # Check similarity with reference answer if available
            if expected_answer and len(expected_answer) > 20:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                try:
                    vectorizer = TfidfVectorizer(stop_words='english')
                    vectors = vectorizer.fit_transform([user_answer.lower(), expected_answer.lower()])
                    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                    
                    # Adjust score based on similarity
                    if similarity > 0.5:
                        # High similarity - boost score
                        score = min(5, base_score + 0.5)
                    elif similarity > 0.3:
                        # Good similarity - slight boost
                        score = min(5, base_score + 0.3)
                    elif similarity > 0.15:
                        # Acceptable similarity - use base
                        score = base_score
                    else:
                        # Low similarity - might be off-topic
                        score = max(2, base_score - 1)
                        if score < 3:
                            feedback_list = evaluator.get_feedback(score, rf_result['features'])
                            feedback_list.insert(0, "⚠️  Your answer may not fully address the question. Review the question carefully.")
                except:
                    score = base_score
            else:
                score = base_score
            
            confidence = rf_result['confidence']
            features = rf_result['features']
            score_10 = score * 2.0
            feedback_list = evaluator.get_feedback(score, features)
        
        feedback = {
            'final_score': score_10,
            'semantic_score': score_10,
            'structure_score': score_10,
            'keyword_score': score_10,
            'improvements': feedback_list,
            'rf_score': score,
            'rf_confidence': confidence if 'rf_result' in locals() else confidence,
            'word_count': word_count
        }
    else:
        # Legacy evaluator (AnswerEvaluator)
        score, feedback = evaluator.evaluate_answer(
            user_answer, 
            expected_answer,
            question.get("keywords", [])
        )
    
    # Compare with human evaluations (only if answer is available and using legacy evaluator)
    human_comparison = None
    if expected_answer and hasattr(evaluator, 'compare_with_human_evaluation'):
        human_comparison = evaluator.compare_with_human_evaluation(
            role, 
            question["question"], 
            expected_answer
        )
    
    # Print detailed feedback
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS (Random Forest AI)")
    print("="*70)
    
    # Show RF-specific metrics if available
    if 'rf_score' in feedback:
        if 'word_count' in feedback:
            print(f"Answer Length       : {feedback['word_count']} words")
        print(f"AI Score (1-5 scale): {feedback['rf_score']}/5 ⭐")
        print(f"Confidence          : {feedback['rf_confidence']:.1%}")
        print(f"Overall Score (10pt): {feedback['final_score']:.2f}/10")
        
        # Show quality indicator
        if feedback['final_score'] < 4:
            print(f"Quality Level       : ❌ Poor - Needs significant improvement")
        elif feedback['final_score'] < 6:
            print(f"Quality Level       : ⚠️  Below Average - Add more details")
        elif feedback['final_score'] < 8:
            print(f"Quality Level       : ✓ Good - Solid answer")
        else:
            print(f"Quality Level       : ⭐ Excellent - Well explained")
    else:
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
    
    # Initialize Random Forest evaluator
    print("\n🤖 Loading Random Forest AI Evaluator...")
    rf_evaluator = RandomForestAnswerEvaluator()
    try:
        rf_evaluator.load_model()
        print("✅ Random Forest model loaded successfully!")
        print("   📊 23 engineered features (STAR, competencies, linguistics)")
        print("   🎯 Trained on 1,514 interview Q&A pairs")
        print("   ⚡ 65% accuracy, 100% within ±1 score")
    except FileNotFoundError:
        print("⚠️  Random Forest model not found. Training new model...")
        rf_evaluator.train_model()
        print("✅ Model training complete!")
    
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
        print("📊 1,470 Q&A pairs across 21 competency categories")
        print("✅ Reference answers available - Similarity checking ENABLED")
        print("📚 Expert STAR format examples with human scores (1-5)")
        print("🎯 You'll answer 3 questions, compared against 100+ references")
        print("💡 Detailed answers required (35+ words for best scores)")
        print("="*70)
        
        all_questions_data = loader.get_interview_questions_dataset(format='json')
    
    elif category == "webdev":
        # Start web development interview
        print("\n" + "="*70)
        print("WEB DEVELOPMENT INTERVIEW")
        print("="*70)
        print("📊 44 Q&A pairs covering modern web technologies")
        print("✅ Reference answers available - Similarity checking ENABLED")
        print("📚 Expert answers scored 8-10/10")
        print("🎯 You'll answer 3 questions, compared against expert answers")
        print("💡 Short correct answers (2-5 words) will be validated via similarity")
        print("="*70)
        
        # Load webdev_interview_qa.csv
        webdev_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'real_dataset_score', 'webdev_interview_qa.csv')
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
    
    elif category in ["python", "java", "csharp", "javascript", "database"]:
        # Load Stack Overflow dataset with tag filtering
        domain_info = {
            'python': {
                'name': 'PYTHON PROGRAMMING',
                'tags': ['python', 'django', 'flask', 'pandas', 'numpy', 'scipy'],
                'count': 1288,
                'desc': 'Python, Django, Flask, Data Science'
            },
            'java': {
                'name': 'JAVA DEVELOPMENT',
                'tags': ['java', 'spring', 'hibernate', 'maven', 'android'],
                'count': 2711,
                'desc': 'Java, Spring, Android, Enterprise'
            },
            'csharp': {
                'name': 'C# / .NET DEVELOPMENT',
                'tags': ['c#', '.net', 'asp.net', 'entity-framework', 'wpf', 'visual-studio'],
                'count': 3022,
                'desc': 'C#, ASP.NET, Entity Framework, WPF'
            },
            'javascript': {
                'name': 'JAVASCRIPT / NODE.JS',
                'tags': ['javascript', 'node.js', 'react', 'angular', 'vue.js', 'typescript'],
                'count': 1026,
                'desc': 'JavaScript, React, Node.js, TypeScript'
            },
            'database': {
                'name': 'DATABASE / SQL',
                'tags': ['sql', 'mysql', 'postgresql', 'mongodb', 'database', 'oracle'],
                'count': 1025,
                'desc': 'SQL, MySQL, PostgreSQL, MongoDB'
            }
        }
        
        info = domain_info[category]
        print("\n" + "="*70)
        print(f"{info['name']} INTERVIEW")
        print("="*70)
        print(f"📊 {info['count']} Stack Overflow Q&A pairs")
        print(f"✅ Topics: {info['desc']}")
        print("✅ Community-validated answers with normalized scores")
        print("🎯 You'll answer 5 questions from real developer problems")
        print("💡 Technical answers validated via similarity matching")
        print("="*70)
        
        # Load stackoverflow_training_data.csv and filter by tags
        so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'real_dataset_score', 'stackoverflow_training_data.csv')
        if os.path.exists(so_path):
            df = pd.read_csv(so_path)
            all_questions_data = []
            
            for _, row in df.iterrows():
                tags_str = str(row['tags']).lower()
                # Check if any of the domain tags are present
                if any(tag in tags_str for tag in info['tags']):
                    all_questions_data.append({
                        'question': row['question'],
                        'answer': row['user_answer'],
                        'competency': row['tags'],
                        'human_score': row['score']
                    })
            
            print(f"✅ Filtered to {len(all_questions_data)} questions for {info['name']}")
        else:
            print("❌ Stack Overflow dataset not found!")
            return
            
    # Removed: ml_ai, software_engineering, deep_learning datasets
    else:
        print("\n❌ Invalid category!")
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
    
    # Determine number of questions based on category
    if category in ["python", "java", "csharp", "javascript", "database"]:
        NUM_QUESTIONS = 5  # More questions for Stack Overflow technical domains
    else:
        NUM_QUESTIONS = 3  # Standard for behavioral/webdev
        
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
        
        if handle_behavioral_answer(question, role, rf_evaluator, all_scores, all_human_comparisons, ref_loader, category):
            attempted += 1

    # Show final summary
    show_session_summary(selected_questions, attempted, all_scores, all_human_comparisons)

if __name__ == "__main__":
    run_interview_session()
