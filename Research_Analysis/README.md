# Research Analysis - Real Dataset Integration

## Overview
The Research_Analysis folder now uses the **SAME production datasets** as the AI Interview Bot application, ensuring consistency between research and deployment.

---

## Dataset Location

### Source (Production)
```
AI_Interview_Bot/data/real_dataset_score/
├── interview_data_with_scores.csv
├── webdev_interview_qa.csv
├── stackoverflow_training_data.csv
├── combined_training_data.csv
└── random_forest_model.joblib
```

### Research Copy
```
Research_Analysis/real_dataset_score/
├── interview_data_with_scores.csv     (Synced)
├── webdev_interview_qa.csv            (Synced)
├── stackoverflow_training_data.csv    (Synced)
├── combined_training_data.csv         (Synced)
└── random_forest_model.joblib         (Synced)
```

---

## Datasets Overview

### 1. interview_data_with_scores.csv
- **Records**: 1,470 Q&A pairs
- **Type**: Behavioral interview questions (STAR format)
- **Columns**: question, answer, competency, human_score
- **Score Range**: 1-5
- **Use Case**: Behavioral interview training and evaluation

### 2. webdev_interview_qa.csv
- **Records**: 44 Q&A pairs
- **Type**: Web development technical questions
- **Columns**: question, answer, competency, human_score
- **Score Range**: 8-10
- **Use Case**: Web development domain interviews

### 3. stackoverflow_training_data.csv
- **Records**: 10,000 Q&A pairs
- **Type**: Stack Overflow technical questions (multiple domains)
- **Columns**: question, user_answer, score, tags, original_score, question_id, answer_id
- **Score Range**: 0.0-5.0 (normalized from original SO scores)
- **Domains**: Python (1,288), Java (2,711), C#/.NET (3,022), JavaScript (1,026), Database (1,025)
- **Use Case**: Technical interview training across multiple programming domains

### 4. combined_training_data.csv
- **Records**: 11,470 Q&A pairs
- **Type**: Combined dataset for model training
- **Columns**: question, user_answer, score
- **Composition**: 10,000 Stack Overflow + 1,470 Behavioral
- **Use Case**: Random Forest model training

### 5. random_forest_model.joblib
- **Type**: Trained scikit-learn RandomForestClassifier
- **Features**: 23 engineered features
- **Training Data**: 11,470 samples
- **Performance**: 65% accuracy, 100% within ±1 score

---

## Analysis Scripts

### analyze_real_datasets.py
Comprehensive analysis script that:
- Loads all 4 datasets
- Analyzes score distributions
- Examines competencies/tags/topics
- Calculates answer length statistics
- Performs data quality checks
- Generates comparison statistics
- Saves analysis results to `outputs/dataset_analysis_summary.json`

**Usage**:
```powershell
cd Research_Analysis
python analyze_real_datasets.py
```

### Untitled-1.ipynb (Updated)
Jupyter notebook with:
- Data loading from real datasets
- Exploratory data analysis (EDA)
- Visualizations
- Statistical analysis
- Model evaluation

**Usage**:
```powershell
cd Research_Analysis
jupyter notebook Untitled-1.ipynb
```

---

## Key Metrics Summary

| Dataset | Records | Avg Score | Avg Words | Unique Categories |
|---------|---------|-----------|-----------|-------------------|
| Behavioral | 1,470 | 2.5 | ~60 | 21 competencies |
| Web Dev | 44 | 8.5 | ~55 | 15 topics |
| Stack Overflow | 10,000 | 2.8 | ~85 | 2,675 tags |
| Combined | 11,470 | 2.7 | ~82 | - |

---

## Data Flow

```
Production Datasets
(AI_Interview_Bot/data/real_dataset_score/)
        ↓
    [SYNCED]
        ↓
Research Datasets
(Research_Analysis/real_dataset_score/)
        ↓
   [ANALYZED]
        ↓
├── analyze_real_datasets.py
├── Untitled-1.ipynb
└── Future Analysis Scripts
```

---

## Synchronization

To keep research datasets in sync with production:

```powershell
# Copy all datasets
Copy-Item -Path "AI_Interview_Bot\data\real_dataset_score\*" `
          -Destination "Research_Analysis\real_dataset_score\" `
          -Force

# Or copy specific files
Copy-Item -Path "AI_Interview_Bot\data\real_dataset_score\interview_data_with_scores.csv" `
          -Destination "Research_Analysis\real_dataset_score\" -Force
```

**Automated sync (PowerShell script)**:
```powershell
# sync_datasets.ps1
$source = "AI_Interview_Bot\data\real_dataset_score\"
$dest = "Research_Analysis\real_dataset_score\"

Write-Host "Syncing datasets..." -ForegroundColor Green
Copy-Item -Path "$source*" -Destination $dest -Force -Recurse
Write-Host "Sync complete!" -ForegroundColor Green
```

---

## Benefits of Unified Datasets

### ✅ Consistency
- Research findings apply directly to production
- No data drift between training and deployment
- Same evaluation criteria

### ✅ Reproducibility
- Experiments can be replicated exactly
- Model performance is verified on same data
- Results are reliable and actionable

### ✅ Efficiency
- No duplicate dataset management
- Single source of truth
- Easy to update and maintain

### ✅ Quality
- Production data is battle-tested
- Real-world Q&A from Stack Overflow
- Human-validated scores

---

## Usage in Research

### Loading Datasets in Python

```python
import pandas as pd

# Load behavioral dataset
behavioral = pd.read_csv('real_dataset_score/interview_data_with_scores.csv')

# Load web dev dataset
webdev = pd.read_csv('real_dataset_score/webdev_interview_qa.csv')

# Load Stack Overflow dataset
stackoverflow = pd.read_csv('real_dataset_score/stackoverflow_training_data.csv')

# Load combined training data
combined = pd.read_csv('real_dataset_score/combined_training_data.csv')

# Load trained model
import joblib
model_data = joblib.load('real_dataset_score/random_forest_model.joblib')
model = model_data['model']
feature_names = model_data['feature_names']
```

### Filtering Stack Overflow by Domain

```python
# Filter Python questions
python_tags = ['python', 'django', 'flask', 'pandas', 'numpy']
python_df = stackoverflow[
    stackoverflow['tags'].str.lower().str.contains('|'.join(python_tags))
]

# Filter Java questions
java_tags = ['java', 'spring', 'hibernate']
java_df = stackoverflow[
    stackoverflow['tags'].str.lower().str.contains('|'.join(java_tags))
]
```

---

## Future Enhancements

### Planned Analysis
1. **Cross-domain comparison**: Compare answer quality across technical domains
2. **Score prediction analysis**: Evaluate Random Forest model performance
3. **Feature importance study**: Which features matter most for scoring
4. **NLTK preprocessing impact**: Measure accuracy gain from lemmatization
5. **Domain adaptation**: How well does model generalize across domains

### Dataset Expansion
- Add more Stack Overflow questions from additional domains
- Include system design questions
- Add algorithm/data structure questions
- Incorporate real interview feedback data

---

## Files Modified

### Updated for Real Datasets
- ✅ `Untitled-1.ipynb` - Updated data loading cells
- ✅ `analyze_real_datasets.py` - Created new comprehensive analysis
- ✅ `README.md` - This documentation

### Data Directory Structure
```
Research_Analysis/
├── data/                           (Legacy/archived)
├── real_dataset_score/             (ACTIVE - Production datasets)
│   ├── interview_data_with_scores.csv
│   ├── webdev_interview_qa.csv
│   ├── stackoverflow_training_data.csv
│   ├── combined_training_data.csv
│   └── random_forest_model.joblib
├── outputs/                        (Analysis results)
│   └── dataset_analysis_summary.json
├── analyze_real_datasets.py        (Analysis script)
├── Untitled-1.ipynb               (Jupyter notebook)
└── README.md                       (This file)
```

---

## Contact & Maintenance

**Last Updated**: November 14, 2025
**Datasets Version**: Production v1.0
**Model Version**: Random Forest v1.0 (65% accuracy)

For questions or issues with datasets, refer to:
- `AI_Interview_Bot/docs_application/DATASETS_DOCUMENTATION.md`
- `AI_Interview_Bot/docs_application/SCORING_FLOW_EXPLAINED.md`
- `AI_Interview_Bot/docs_application/NLTK_TEXT_PREPROCESSING.md`
