"""
AI vs Human Evaluator - Compare AI predictions with human assessor scores
Based on research paper methodology for competency assessment
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix
import json
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class CompetencyEvaluator:
    """
    Advanced AI evaluator that mimics human assessor scoring
    Uses research paper's BEI methodology with TF-IDF and keyword matching
    """
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        
        # Competency keywords based on STAR methodology
        self.competency_keywords = {
            'leadership': {
                'level_1': ['help', 'assist', 'support', 'follow'],
                'level_2': ['coordinate', 'organize', 'plan', 'manage', 'team'],
                'level_3': ['lead', 'direct', 'guide', 'mentor', 'supervise', 'strategic'],
                'level_4': ['vision', 'inspire', 'transform', 'innovate', 'delegate', 'empower'],
                'level_5': ['pioneer', 'revolutionize', 'architect', 'champion', 'visionary']
            },
            'teamwork': {
                'level_1': ['participate', 'attend', 'join', 'collaborate'],
                'level_2': ['contribute', 'share', 'communicate', 'cooperate', 'support'],
                'level_3': ['facilitate', 'coordinate', 'integrate', 'mediate', 'align'],
                'level_4': ['build', 'foster', 'strengthen', 'cultivate', 'synergize'],
                'level_5': ['transform', 'architect', 'champion', 'unite', 'harmonize']
            },
            'integrity': {
                'level_1': ['honest', 'truthful', 'follow', 'obey', 'comply'],
                'level_2': ['ethical', 'responsible', 'accountable', 'transparent', 'rules'],
                'level_3': ['principle', 'values', 'standards', 'uphold', 'enforce'],
                'level_4': ['exemplify', 'model', 'champion', 'advocate', 'embed'],
                'level_5': ['institutionalize', 'transform', 'legacy', 'culture', 'paradigm']
            },
            'communication': {
                'level_1': ['tell', 'inform', 'say', 'report', 'listen'],
                'level_2': ['explain', 'clarify', 'discuss', 'present', 'articulate'],
                'level_3': ['persuade', 'influence', 'negotiate', 'facilitate', 'engage'],
                'level_4': ['inspire', 'compelling', 'diplomatic', 'strategic', 'resonate'],
                'level_5': ['transform', 'visionary', 'paradigm', 'profound', 'revolutionary']
            },
            'result_oriented': {
                'level_1': ['complete', 'finish', 'do', 'task', 'work'],
                'level_2': ['achieve', 'target', 'goal', 'meet', 'deliver', 'improve'],
                'level_3': ['exceed', 'optimize', 'enhance', 'measure', 'track', 'efficiency'],
                'level_4': ['strategic', 'impact', 'innovative', 'breakthrough', 'excellence'],
                'level_5': ['transform', 'revolutionize', 'industry-leading', 'benchmark', 'paradigm']
            },
            'problem_solving': {
                'level_1': ['identify', 'recognize', 'find', 'notice', 'issue'],
                'level_2': ['analyze', 'investigate', 'examine', 'assess', 'evaluate'],
                'level_3': ['solve', 'resolve', 'implement', 'develop', 'design', 'systematic'],
                'level_4': ['innovate', 'strategic', 'complex', 'creative', 'breakthrough'],
                'level_5': ['pioneer', 'revolutionary', 'paradigm', 'architect', 'transform']
            },
            'people_development': {
                'level_1': ['help', 'assist', 'support', 'teach', 'show'],
                'level_2': ['train', 'coach', 'guide', 'mentor', 'develop'],
                'level_3': ['empower', 'cultivate', 'nurture', 'build', 'strengthen'],
                'level_4': ['transform', 'inspire', 'elevate', 'strategic', 'succession'],
                'level_5': ['legacy', 'institutional', 'culture', 'paradigm', 'revolutionary']
            }
        }
        
        # STAR structure keywords
        self.star_keywords = {
            'situation': ['faced', 'encountered', 'situation', 'problem', 'challenge', 'scenario', 
                         'when', 'during', 'time', 'experience', 'context'],
            'task': ['responsible', 'assigned', 'task', 'objective', 'goal', 'required', 'needed',
                    'role', 'duty', 'expected', 'mission'],
            'action': ['did', 'implemented', 'executed', 'performed', 'managed', 'led', 'organized',
                      'created', 'developed', 'improved', 'coordinated', 'initiated', 'established'],
            'result': ['achieved', 'resulted', 'outcome', 'impact', 'success', 'improved', 'increased',
                      'decreased', 'delivered', 'completed', 'exceeded', 'accomplished']
        }
        
        # POS tagging weights (verbs get highest weight as per research paper)
        self.pos_weights = {
            'verb': 1.5,
            'noun': 1.0,
            'adjective': 0.8,
            'adverb': 0.7,
            'other': 0.5
        }
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text or pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def calculate_star_score(self, answer):
        """Calculate STAR structure completeness (0-4)"""
        answer_lower = self.preprocess_text(answer)
        star_score = 0
        star_components = {}
        
        for component, keywords in self.star_keywords.items():
            found = any(keyword in answer_lower for keyword in keywords)
            star_components[component] = found
            if found:
                star_score += 1
        
        return star_score, star_components
    
    def calculate_tf_score(self, answer, competency):
        """
        Calculate TF-based competency score using research paper methodology
        Formula: TF = (Number of times word appears) / (Total words in level)
        """
        answer_lower = self.preprocess_text(answer)
        words = answer_lower.split()
        
        if not words:
            return 0, 0
        
        # Get competency keywords
        comp_key = competency.lower().replace(' ', '_')
        if comp_key not in self.competency_keywords:
            comp_key = 'result_oriented'  # Default
        
        keyword_dict = self.competency_keywords[comp_key]
        
        # Calculate TF score for each level
        level_scores = {}
        for level_name, keywords in keyword_dict.items():
            level_num = int(level_name.split('_')[1])
            
            # Count keyword matches
            match_count = sum(1 for word in words if word in keywords)
            
            # Calculate TF (as per research paper Equation 1)
            tf_score = match_count / len(words) if len(words) > 0 else 0
            
            # Apply verb weighting (POS tagging simulation)
            # Higher weight for action words
            action_words = [w for w in words if w in self.star_keywords['action']]
            verb_boost = len(action_words) * 0.1
            
            level_scores[level_num] = tf_score + verb_boost
        
        # Get maximum score level (as per research paper Equation 3)
        if not level_scores or max(level_scores.values()) == 0:
            return 0, level_scores  # All-zero result condition
        
        predicted_level = max(level_scores, key=level_scores.get)
        max_score = level_scores[predicted_level]
        
        return predicted_level, level_scores
    
    def calculate_length_score(self, answer):
        """Calculate answer quality based on length and structure"""
        words = answer.split()
        word_count = len(words)
        
        # Optimal range: 50-150 words
        if 50 <= word_count <= 150:
            return 1.0
        elif 25 <= word_count < 50:
            return 0.7
        elif 150 < word_count <= 200:
            return 0.8
        elif word_count < 25:
            return 0.3
        else:
            return 0.6
    
    def evaluate_answer(self, answer, competency, human_score=None):
        """
        Main evaluation function - combines all scoring methods
        Returns AI predicted score (0-5 scale)
        """
        if not answer or pd.isna(answer) or len(str(answer).strip()) < 10:
            return {
                'ai_score': 0,
                'star_score': 0,
                'tf_level': 0,
                'length_quality': 0,
                'human_score': human_score,
                'is_interpretable': False
            }
        
        answer = self.preprocess_text(answer)
        
        # 1. STAR structure analysis (40% weight)
        star_score, star_components = self.calculate_star_score(answer)
        star_normalized = star_score / 4  # Normalize to 0-1
        
        # 2. TF-based competency level (40% weight)
        tf_level, level_scores = self.calculate_tf_score(answer, competency)
        tf_normalized = tf_level / 5  # Normalize to 0-1
        
        # 3. Length and quality (20% weight)
        length_quality = self.calculate_length_score(answer)
        
        # Combined score (weighted average)
        ai_score_normalized = (star_normalized * 0.4) + (tf_normalized * 0.4) + (length_quality * 0.2)
        
        # Convert to 0-5 scale (matching human scoring)
        ai_score = round(ai_score_normalized * 5)
        
        # All-zero check
        is_interpretable = ai_score > 0
        
        return {
            'ai_score': ai_score,
            'star_score': star_score,
            'tf_level': tf_level,
            'length_quality': length_quality,
            'star_components': star_components,
            'level_scores': level_scores,
            'human_score': human_score,
            'is_interpretable': is_interpretable
        }


class ModelPerformanceAnalyzer:
    """Analyze AI model performance against human scores"""
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.results = []
    
    def load_human_scores(self, csv_path):
        """Load human assessor scores from CSV"""
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(df)} records from {csv_path}")
            print(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None
    
    def evaluate_dataset(self, df):
        """Evaluate entire dataset with AI and compare to human scores"""
        results = []
        
        required_cols = ['question', 'answer', 'competency', 'human_score']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️ Missing columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        for idx, row in df.iterrows():
            result = self.evaluator.evaluate_answer(
                answer=row['answer'],
                competency=row['competency'],
                human_score=row['human_score']
            )
            
            result['question'] = row['question']
            result['competency'] = row['competency']
            result['employee_id'] = row.get('employee_id', idx)
            
            # Calculate difference
            if result['is_interpretable']:
                result['difference'] = result['ai_score'] - result['human_score']
                result['absolute_error'] = abs(result['difference'])
                result['status'] = 'Accurate' if result['difference'] == 0 else 'Inaccurate'
            else:
                result['difference'] = None
                result['absolute_error'] = None
                result['status'] = 'Uninterpretable'
            
            results.append(result)
        
        self.results = results
        return pd.DataFrame(results)
    
    def calculate_metrics(self, results_df):
        """Calculate performance metrics: Coverage, Accuracy, MAE, etc."""
        total = len(results_df)
        interpretable = results_df['is_interpretable'].sum()
        uninterpretable = total - interpretable
        
        # Coverage (as per research paper Equation 4)
        coverage = (interpretable / total) * 100 if total > 0 else 0
        
        # Filter interpretable results
        valid_results = results_df[results_df['is_interpretable'] == True]
        
        if len(valid_results) == 0:
            return {
                'total_records': total,
                'interpretable': interpretable,
                'uninterpretable': uninterpretable,
                'coverage': coverage,
                'accuracy': 0,
                'mae': 0,
                'rmse': 0,
                'exact_match': 0
            }
        
        # Accuracy - percentage of exact matches
        exact_matches = (valid_results['ai_score'] == valid_results['human_score']).sum()
        accuracy = (exact_matches / len(valid_results)) * 100
        
        # Mean Absolute Error
        mae = valid_results['absolute_error'].mean()
        
        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(valid_results['human_score'], valid_results['ai_score']))
        
        # Within 1 point accuracy (tolerance)
        within_1 = (valid_results['absolute_error'] <= 1).sum()
        within_1_accuracy = (within_1 / len(valid_results)) * 100
        
        return {
            'total_records': total,
            'interpretable': interpretable,
            'uninterpretable': uninterpretable,
            'coverage': coverage,
            'accuracy': accuracy,
            'within_1_accuracy': within_1_accuracy,
            'mae': mae,
            'rmse': rmse,
            'exact_match': exact_matches
        }
    
    def generate_confusion_matrix(self, results_df):
        """Generate confusion matrix for visualization"""
        valid_results = results_df[results_df['is_interpretable'] == True]
        
        if len(valid_results) == 0:
            return None
        
        cm = confusion_matrix(
            valid_results['human_score'], 
            valid_results['ai_score'],
            labels=[0, 1, 2, 3, 4, 5]
        )
        
        return cm
    
    def visualize_performance(self, results_df, metrics, output_dir):
        """Create visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        valid_results = results_df[results_df['is_interpretable'] == True]
        
        if len(valid_results) == 0:
            print("⚠️ No interpretable results to visualize")
            return
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = self.generate_confusion_matrix(results_df)
        if cm is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=[0,1,2,3,4,5], yticklabels=[0,1,2,3,4,5])
            plt.title('AI vs Human Score Confusion Matrix', fontsize=14, fontweight='bold')
            plt.xlabel('AI Predicted Score', fontsize=12)
            plt.ylabel('Human Assessor Score', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
            plt.close()
        
        # 2. Score Distribution Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(valid_results['human_score'], bins=6, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_title('Human Score Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Score')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(valid_results['ai_score'], bins=6, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_title('AI Score Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Score')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=300)
        plt.close()
        
        # 3. Scatter plot: AI vs Human
        plt.figure(figsize=(10, 8))
        plt.scatter(valid_results['human_score'], valid_results['ai_score'], 
                   alpha=0.6, s=100, c='purple', edgecolors='black')
        plt.plot([0, 5], [0, 5], 'r--', linewidth=2, label='Perfect Agreement')
        plt.xlabel('Human Assessor Score', fontsize=12)
        plt.ylabel('AI Predicted Score', fontsize=12)
        plt.title('AI vs Human Score Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ai_vs_human_scatter.png'), dpi=300)
        plt.close()
        
        # 4. Performance Metrics Bar Chart
        plt.figure(figsize=(10, 6))
        metrics_to_plot = ['coverage', 'accuracy', 'within_1_accuracy']
        values = [metrics[m] for m in metrics_to_plot]
        colors = ['#3498db', '#2ecc71', '#f39c12']
        
        bars = plt.bar(metrics_to_plot, values, color=colors, edgecolor='black', linewidth=1.5)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.title('AI Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylim(0, 105)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
        plt.xticks(['coverage', 'accuracy', 'within_1_accuracy'], 
                  ['Coverage', 'Exact Match\nAccuracy', 'Within ±1\nAccuracy'])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300)
        plt.close()
        
        print(f"\n✅ Visualizations saved to: {output_dir}")
    
    def save_results(self, results_df, metrics, output_dir):
        """Save detailed results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results CSV
        results_path = os.path.join(output_dir, 'ai_vs_human_detailed_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"✅ Detailed results saved: {results_path}")
        
        # Save metrics JSON (convert numpy types to Python types)
        metrics_path = os.path.join(output_dir, 'performance_metrics.json')
        metrics_serializable = {k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v) if isinstance(v, (int, np.int64)) else v 
                               for k, v in metrics.items()}
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"✅ Metrics saved: {metrics_path}")
        
        # Generate summary report
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("AI vs HUMAN EVALUATOR - PERFORMANCE REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Records: {metrics['total_records']}\n")
            f.write(f"Interpretable Records: {metrics['interpretable']}\n")
            f.write(f"Uninterpretable Records: {metrics['uninterpretable']}\n\n")
            
            f.write(f"Coverage: {metrics['coverage']:.2f}%\n")
            f.write(f"Exact Match Accuracy: {metrics['accuracy']:.2f}%\n")
            f.write(f"Within ±1 Accuracy: {metrics['within_1_accuracy']:.2f}%\n")
            f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.3f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.3f}\n\n")
            
            # Competency-wise breakdown
            f.write("="*70 + "\n")
            f.write("COMPETENCY-WISE PERFORMANCE\n")
            f.write("="*70 + "\n\n")
            
            valid_results = results_df[results_df['is_interpretable'] == True]
            for comp in valid_results['competency'].unique():
                comp_data = valid_results[valid_results['competency'] == comp]
                comp_acc = ((comp_data['ai_score'] == comp_data['human_score']).sum() / len(comp_data)) * 100
                comp_mae = comp_data['absolute_error'].mean()
                
                f.write(f"{comp}:\n")
                f.write(f"  - Records: {len(comp_data)}\n")
                f.write(f"  - Accuracy: {comp_acc:.2f}%\n")
                f.write(f"  - MAE: {comp_mae:.3f}\n\n")
        
        print(f"✅ Summary report saved: {report_path}")


def create_sample_dataset():
    """Create a sample dataset for testing"""
    data = {
        'employee_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'competency': ['Leadership', 'Teamwork', 'Integrity', 'Communication', 'Result Oriented',
                      'Problem Solving', 'Leadership', 'Teamwork', 'Result Oriented', 'Communication'],
        'question': [
            'Tell me about a time you led a challenging project.',
            'Describe a situation where you worked with a difficult team member.',
            'Give an example of when you had to make an ethical decision.',
            'Tell me about a time you had to present complex information.',
            'Describe how you exceeded targets in your previous role.',
            'Tell me about a complex problem you solved.',
            'Describe your leadership style with examples.',
            'How do you handle conflicts in a team?',
            'What strategies do you use to achieve results?',
            'Give an example of effective communication.'
        ],
        'answer': [
            'In my previous role as project manager, I faced a critical deadline for software launch. I was responsible for coordinating a team of 10 developers. I implemented daily stand-ups, created detailed timelines, and established clear communication channels. As a result, we delivered the project 2 weeks ahead of schedule with 100% client satisfaction.',
            'I worked with a colleague who often missed deadlines. I scheduled a one-on-one meeting to understand their challenges. I helped them organize their tasks and offered support. We collaborated on a shared timeline and I provided regular check-ins. Eventually, they improved their performance significantly.',
            'When I noticed accounting irregularities, I faced pressure to overlook them. I reported the issue to my supervisor following company policy. I documented everything thoroughly and cooperated with the investigation. The issue was resolved, and the company strengthened its controls.',
            'I had to present quarterly financial results to non-technical stakeholders. I created visual charts and simplified complex data. I used analogies to explain technical concepts and allowed time for questions. The presentation was well-received and stakeholders made informed decisions.',
            'My target was 50 sales per month, but I achieved 75 by analyzing customer data, identifying high-potential leads, and personalizing my approach. I also improved my follow-up process and built stronger relationships with clients. This resulted in 50% above target performance.',
            'Our production line had recurring quality issues. I analyzed the entire process, identified the root cause in material handling, and designed a new quality control checkpoint. I trained staff on the new procedure and monitored results. Defects decreased by 80%.',
            'I lead by example and empower my team. I set clear goals and provide resources. I mentor team members and encourage innovation. I celebrate successes and learn from failures together.',
            'I listen to both sides and find common ground. I focus on solutions rather than blame.',
            'I set clear goals, track progress, and adjust strategies as needed.',
            'I communicate clearly and ensure understanding through follow-up.'
        ],
        'human_score': [4, 3, 3, 4, 4, 4, 2, 1, 2, 1]
    }
    
    df = pd.DataFrame(data)
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'data', 'sample_interview_dataset.csv')
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    df.to_csv(sample_path, index=False)
    print(f"✅ Sample dataset created: {sample_path}")
    return sample_path


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("AI vs HUMAN EVALUATOR - COMPETENCY ASSESSMENT SYSTEM")
    print("="*70 + "\n")
    
    # Initialize evaluator
    evaluator = CompetencyEvaluator()
    analyzer = ModelPerformanceAnalyzer(evaluator)
    
    # Create sample dataset (or use your existing CSV)
    csv_path = create_sample_dataset()
    
    # You can replace this with your actual dataset path:
    # csv_path = "path/to/your/interview_data.csv"
    
    print("\n📊 Loading human assessor scores...")
    df = analyzer.load_human_scores(csv_path)
    
    if df is None:
        print("❌ Failed to load dataset")
        return
    
    print(f"\n🤖 Evaluating {len(df)} answers with AI...")
    results_df = analyzer.evaluate_dataset(df)
    
    if results_df is None:
        print("❌ Evaluation failed")
        return
    
    print("\n📈 Calculating performance metrics...")
    metrics = analyzer.calculate_metrics(results_df)
    
    # Print results
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    print(f"Total Records:            {metrics['total_records']}")
    print(f"Interpretable:            {metrics['interpretable']}")
    print(f"Uninterpretable:          {metrics['uninterpretable']}")
    print(f"\nCoverage:                 {metrics['coverage']:.2f}%")
    print(f"Exact Match Accuracy:     {metrics['accuracy']:.2f}%")
    print(f"Within ±1 Accuracy:       {metrics['within_1_accuracy']:.2f}%")
    print(f"Mean Absolute Error:      {metrics['mae']:.3f}")
    print(f"Root Mean Squared Error:  {metrics['rmse']:.3f}")
    print("="*70)
    
    # Show sample comparisons
    print("\n" + "="*70)
    print("SAMPLE COMPARISONS (First 5 records)")
    print("="*70)
    valid_results = results_df[results_df['is_interpretable'] == True].head(5)
    for idx, row in valid_results.iterrows():
        print(f"\nEmployee {row['employee_id']} - {row['competency']}")
        print(f"  Human Score: {row['human_score']}")
        print(f"  AI Score:    {row['ai_score']}")
        print(f"  Difference:  {row['difference']:+.0f}")
        print(f"  Status:      {row['status']}")
    
    # Save results and visualizations
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'data', 'ai_evaluation_results')
    
    print(f"\n💾 Saving results to: {output_dir}")
    analyzer.save_results(results_df, metrics, output_dir)
    
    print(f"\n📊 Creating visualizations...")
    analyzer.visualize_performance(results_df, metrics, output_dir)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("Files created:")
    print("  - ai_vs_human_detailed_results.csv")
    print("  - performance_metrics.json")
    print("  - evaluation_report.txt")
    print("  - confusion_matrix.png")
    print("  - score_distribution.png")
    print("  - ai_vs_human_scatter.png")
    print("  - performance_metrics.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
