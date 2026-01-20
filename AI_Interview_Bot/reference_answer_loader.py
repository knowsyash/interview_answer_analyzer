"""
Reference Answer Loader
Loads reference answers from interview_data_with_scores.csv and provides comparison functionality
"""

import pandas as pd
import json
import os
from typing import Dict, List, Optional


class ReferenceAnswerLoader:
    """Loads and manages reference answers from CSV dataset"""
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try multiple possible locations
        possible_dirs = [
            os.path.join(self.script_dir, 'real_dataset_score'),
            os.path.join(self.script_dir, '..', 'real_dataset_score')
        ]
        self.data_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                self.data_dir = dir_path
                break
        if not self.data_dir:
            self.data_dir = possible_dirs[0]  # Default to first option
        self.reference_file = os.path.join(self.data_dir, 'interview_data_with_scores.csv')
        self.reference_data = None
        self.competency_answers = {}
        
    def load_reference_answers(self) -> bool:
        """Load reference answers from CSV file"""
        try:
            if not os.path.exists(self.reference_file):
                print(f"⚠️ Reference file not found: {self.reference_file}")
                return False
                
            # Load CSV
            self.reference_data = pd.read_csv(self.reference_file)
            print(f"✅ Loaded {len(self.reference_data)} reference Q&A pairs")
            
            # Organize by competency
            self._organize_by_competency()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading reference answers: {e}")
            return False
    
    def _organize_by_competency(self):
        """Organize reference answers by competency for quick lookup"""
        for idx, row in self.reference_data.iterrows():
            try:
                # Parse competency (it's stored as string representation of list)
                competency_str = row['competency']
                if isinstance(competency_str, str):
                    # Remove brackets and quotes, split by comma
                    competencies = [c.strip().strip("'\"[]") for c in competency_str.split(',')]
                else:
                    competencies = [str(competency_str)]
                
                # Add to each competency category
                for comp in competencies:
                    comp = comp.strip()
                    if comp not in self.competency_answers:
                        self.competency_answers[comp] = []
                    
                    self.competency_answers[comp].append({
                        'question': row['question'],
                        'answer': row['answer'],
                        'human_score': row['human_score'],
                        'competency': competencies
                    })
            except Exception as e:
                continue
        
        print(f"📊 Organized into {len(self.competency_answers)} competency categories")
    
    def get_reference_answer(self, question: str, competency: Optional[str] = None) -> Optional[Dict]:
        """
        Get the best matching reference answer for a question
        
        Args:
            question: The question being asked
            competency: Optional competency to narrow search
            
        Returns:
            Dictionary with reference answer details or None
        """
        if self.reference_data is None:
            return None
        
        # Try to find exact or close match
        matches = []
        
        if competency and competency in self.competency_answers:
            # Search within competency
            search_pool = self.competency_answers[competency]
        else:
            # Search all reference data
            search_pool = self.reference_data.to_dict('records')
        
        # Find questions with similar keywords
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        for ref in search_pool:
            ref_question = ref['question'].lower() if 'question' in ref else ''
            ref_words = set(ref_question.split())
            
            # Calculate word overlap
            overlap = len(question_words & ref_words)
            if overlap > 3:  # At least 3 common words
                matches.append((overlap, ref))
        
        if matches:
            # Return best match (highest overlap)
            matches.sort(reverse=True, key=lambda x: x[0])
            best_match = matches[0][1]
            
            return {
                'question': best_match.get('question', ''),
                'answer': best_match.get('answer', ''),
                'human_score': best_match.get('human_score', 0),
                'competency': best_match.get('competency', [])
            }
        
        # If no good match, return a random high-scoring answer from same competency
        if competency and competency in self.competency_answers:
            high_scorers = [a for a in self.competency_answers[competency] 
                          if a['human_score'] >= 7]
            if high_scorers:
                return high_scorers[0]
        
        return None
    
    def get_competencies(self) -> List[str]:
        """Get list of all competencies in reference data"""
        return list(self.competency_answers.keys())
    
    def save_to_json(self, output_file: str = None):
        """Save reference answers to JSON format"""
        if output_file is None:
            output_file = os.path.join(self.data_dir, 'reference_answers.json')
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.competency_answers, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved reference answers to {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving to JSON: {e}")
            return False


# Test the loader
if __name__ == "__main__":
    loader = ReferenceAnswerLoader()
    if loader.load_reference_answers():
        print(f"\n📋 Available Competencies: {loader.get_competencies()[:10]}...")
        
        # Test getting a reference answer
        ref = loader.get_reference_answer(
            "Tell me about a time you demonstrated leadership",
            competency="Leadership"
        )
        if ref:
            print(f"\n📝 Sample Reference Answer:")
            print(f"Question: {ref['question'][:100]}...")
            print(f"Answer: {ref['answer'][:150]}...")
            print(f"Human Score: {ref['human_score']}")
        
        # Save to JSON
        loader.save_to_json()
