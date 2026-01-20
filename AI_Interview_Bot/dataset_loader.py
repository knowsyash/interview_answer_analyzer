"""
Comprehensive Dataset Loader
Finds all available datasets and converts them to JSON format when needed
"""

import os
import json
import pandas as pd
from pathlib import Path


class DatasetLoader:
    """
    Smart dataset loader that:
    1. Scans all folders for available datasets
    2. Identifies the correct dataset for each purpose
    3. Converts CSV to JSON on-demand
    4. Caches converted data for performance
    """
    
    def __init__(self):
        # Get the directory where this script is located (AI_Interview_Bot folder)
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        # Try multiple data directory locations for flexibility
        possible_data_dirs = [
            os.path.join(self.project_root, 'real_dataset_score'),
            os.path.join(self.project_root, '..', 'real_dataset_score'),
            os.path.join(self.project_root, 'data')
        ]
        self.data_dir = None
        for dir_path in possible_data_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                self.data_dir = dir_path
                break
        if not self.data_dir:
            self.data_dir = possible_data_dirs[0]  # Default to first option
        self.cache = {}
        self.available_datasets = self._scan_all_datasets()
    
    def _scan_all_datasets(self):
        """Scan all folders and catalog available datasets"""
        datasets = {
            'csv': [],
            'json': []
        }
        
        print("🔍 Scanning for available datasets...")
        
        # Walk through all directories
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.data_dir)
                
                if file.endswith('.csv'):
                    datasets['csv'].append({
                        'name': file,
                        'path': file_path,
                        'relative_path': rel_path,
                        'folder': os.path.basename(root)
                    })
                elif file.endswith('.json'):
                    datasets['json'].append({
                        'name': file,
                        'path': file_path,
                        'relative_path': rel_path,
                        'folder': os.path.basename(root)
                    })
        
        print(f"✅ Found {len(datasets['csv'])} CSV files")
        print(f"✅ Found {len(datasets['json'])} JSON files")
        
        return datasets
    
    def list_available_datasets(self):
        """List all available datasets with their purpose"""
        print("\n" + "="*70)
        print("📊 AVAILABLE DATASETS")
        print("="*70)
        
        print("\n📁 CSV Datasets:")
        for i, ds in enumerate(self.available_datasets['csv'], 1):
            print(f"  {i}. {ds['name']:<40} [{ds['folder']}]")
        
        print("\n📁 JSON Datasets:")
        for i, ds in enumerate(self.available_datasets['json'], 1):
            print(f"  {i}. {ds['name']:<40} [{ds['folder']}]")
        
        return self.available_datasets
    
    def get_interview_questions_dataset(self, format='json'):
        """
        Get the best available interview questions dataset
        
        Priority:
        1. interview_data_with_scores.csv (1,470 Q&A with human scores)
        2. interview_qa_dataset.csv (STAR format)
        3. deeplearning_questions.csv (Questions only)
        4. processed_questions.json (Processed format)
        """
        
        # Check cache first
        cache_key = f'interview_questions_{format}'
        if cache_key in self.cache:
            print("✅ Loading from cache...")
            return self.cache[cache_key]
        
        print("\n🔍 Finding best interview questions dataset...")
        
        # Priority order
        preferred_files = [
            'interview_data_with_scores.csv',
            'interview_qa_dataset.csv',
            'deeplearning_questions.csv'
        ]
        
        for filename in preferred_files:
            dataset = self._find_dataset(filename)
            if dataset:
                print(f"✅ Found: {filename}")
                data = self._load_and_convert(dataset['path'], format)
                self.cache[cache_key] = data
                return data
        
        # Fallback to JSON if exists
        json_file = self._find_dataset('processed_questions.json')
        if json_file:
            print(f"✅ Using fallback: processed_questions.json")
            with open(json_file['path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.cache[cache_key] = data
                return data
        
        print("❌ No interview questions dataset found!")
        return None
    
    def get_human_scores_dataset(self, format='json'):
        """
        Get dataset with human scores for training/evaluation
        
        Best: interview_data_with_scores.csv (1,470 records with scores)
        """
        cache_key = f'human_scores_{format}'
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        print("\n🔍 Finding human scores dataset...")
        
        dataset = self._find_dataset('interview_data_with_scores.csv')
        if dataset:
            print(f"✅ Found: interview_data_with_scores.csv (1,470 Q&A with scores)")
            data = self._load_and_convert(dataset['path'], format)
            self.cache[cache_key] = data
            return data
        
        # Fallback
        dataset = self._find_dataset('sample_interview_dataset.csv')
        if dataset:
            print(f"✅ Using fallback: sample_interview_dataset.csv")
            data = self._load_and_convert(dataset['path'], format)
            self.cache[cache_key] = data
            return data
        
        return None
    
    def get_competency_data(self, format='json'):
        """Get competency mappings and weights"""
        cache_key = f'competency_{format}'
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        print("\n🔍 Finding competency data...")
        
        # Try JSON first
        comp_dict = self._find_dataset('competency_dictionary.json')
        comp_weights = self._find_dataset('competency_weights.json')
        
        if comp_dict and comp_weights:
            with open(comp_dict['path'], 'r') as f1, open(comp_weights['path'], 'r') as f2:
                data = {
                    'dictionary': json.load(f1),
                    'weights': json.load(f2)
                }
                self.cache[cache_key] = data
                return data
        
        # Try CSV
        comp_csv = self._find_dataset('competency_mapping.csv')
        if comp_csv:
            data = self._load_and_convert(comp_csv['path'], format)
            self.cache[cache_key] = data
            return data
        
        return None
    
    def _find_dataset(self, filename):
        """Find a dataset by filename"""
        # Check CSV
        for ds in self.available_datasets['csv']:
            if ds['name'] == filename:
                return ds
        
        # Check JSON
        for ds in self.available_datasets['json']:
            if ds['name'] == filename:
                return ds
        
        return None
    
    def _load_and_convert(self, file_path, target_format='json'):
        """Load CSV and convert to JSON if needed"""
        
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif file_path.endswith('.csv'):
            print(f"📄 Loading CSV: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            
            print(f"   Records: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            
            if target_format == 'json':
                # Convert to JSON format
                data = df.to_dict('records')
                
                # Optionally save as JSON for future use
                json_path = file_path.replace('.csv', '_converted.json')
                print(f"💾 Saving converted JSON to: {os.path.basename(json_path)}")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                return data
            
            elif target_format == 'dataframe':
                return df
            
            else:
                return df.to_dict('records')
        
        return None
    
    def convert_csv_to_json(self, csv_filename, output_filename=None):
        """Manually convert a specific CSV to JSON"""
        
        dataset = self._find_dataset(csv_filename)
        if not dataset:
            print(f"❌ Dataset not found: {csv_filename}")
            return None
        
        print(f"\n🔄 Converting {csv_filename} to JSON...")
        
        # Load CSV
        df = pd.read_csv(dataset['path'])
        data = df.to_dict('records')
        
        # Determine output path
        if output_filename:
            output_path = os.path.join(self.data_dir, output_filename)
        else:
            output_path = dataset['path'].replace('.csv', '_converted.json')
        
        # Save JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved to: {os.path.basename(output_path)}")
        print(f"   Records: {len(data)}")
        
        return output_path
    
    def get_dataset_info(self, filename):
        """Get detailed information about a dataset"""
        dataset = self._find_dataset(filename)
        if not dataset:
            print(f"❌ Dataset not found: {filename}")
            return None
        
        print(f"\n📊 Dataset Info: {filename}")
        print("="*70)
        
        if dataset['path'].endswith('.csv'):
            df = pd.read_csv(dataset['path'])
            print(f"Type: CSV")
            print(f"Records: {len(df)}")
            print(f"Columns: {len(df.columns)}")
            print(f"\nColumn Names:")
            for col in df.columns:
                print(f"  - {col}")
            print(f"\nFirst 3 records:")
            print(df.head(3))
            
        elif dataset['path'].endswith('.json'):
            with open(dataset['path'], 'r') as f:
                data = json.load(f)
            print(f"Type: JSON")
            if isinstance(data, list):
                print(f"Records: {len(data)}")
                if len(data) > 0:
                    print(f"Keys: {list(data[0].keys())}")
            elif isinstance(data, dict):
                print(f"Keys: {list(data.keys())}")
        
        return dataset


def main():
    """Demo usage"""
    loader = DatasetLoader()
    
    # List all available datasets
    loader.list_available_datasets()
    
    # Get interview questions
    print("\n" + "="*70)
    questions = loader.get_interview_questions_dataset(format='json')
    if questions:
        print(f"\n✅ Loaded {len(questions)} interview questions")
        print(f"Sample: {questions[0] if questions else 'None'}")
    
    # Get human scores dataset
    print("\n" + "="*70)
    scores_data = loader.get_human_scores_dataset(format='json')
    if scores_data:
        print(f"\n✅ Loaded {len(scores_data)} records with human scores")
    
    # Get competency data
    print("\n" + "="*70)
    comp_data = loader.get_competency_data()
    if comp_data:
        print(f"\n✅ Loaded competency data")
    
    print("\n" + "="*70)
    print("✅ Dataset Loader Ready!")
    print("="*70)


if __name__ == "__main__":
    main()
