# AI Interview Coach Bot - File Structure

## 📁 Active Workflow Files (Core System)

### Main Application
- **`src/main.py`** - Main entry point for the interview bot
  - Handles interview flow (Technical & Behavioral modes)
  - Integrates TF-IDF scoring and behavioral evaluation
  - Manages question selection and session summary

### Core Components
- **`src/dataset_loader.py`** - Smart dataset management system
  - Auto-scans all data folders for CSV/JSON files
  - Converts CSV to JSON on-demand with caching
  - No hardcoded paths - dynamic discovery

- **`src/tfidf_evaluator.py`** - TF-IDF based answer evaluator (NEW)
  - Uses NLTK for preprocessing (tokenization, lemmatization, stop words)
  - Computes TF-IDF vectors for semantic similarity
  - Scores answers on 0-10 scale (length + relevance + depth)

- **`src/evaluator.py`** - Behavioral answer evaluator
  - Uses sklearn TF-IDF and cosine similarity
  - Evaluates STAR format responses
  - Compares with human expert scores

- **`src/logger.py`** - Session logging system
  - Logs questions, answers, scores, and feedback
  - Saves to `logs/session_log.txt`

- **`src/resources.py`** - Tips and resources
  - Provides interview tips and suggestions

- **`src/evaluate_model.py`** - Model evaluation utilities
  - InterviewBotEvaluator class for model assessment

---

## 📊 Research & Analysis Files (Keep for Reference)

### Data Analysis
- **`src/data_preprocessing_eda.py`** - Exploratory Data Analysis
  - Data preprocessing and visualization
  - Statistical analysis of interview datasets
  - Feature engineering experiments

- **`src/process_hr_data.py`** - HR data processing
  - Processes HR analytics datasets
  - Data cleaning and transformation

- **`src/process_kaggle_data.py`** - Kaggle data processing
  - Processes Kaggle competition datasets
  - Data normalization and preparation

### Model Research & Testing
- **`src/ai_vs_human_evaluator.py`** - AI vs Human comparison
  - Compares AI scoring with human expert scores
  - Generates accuracy metrics and visualizations

- **`src/improved_ai_evaluator.py`** - Enhanced AI evaluator
  - Advanced evaluation techniques
  - Machine learning-based scoring improvements

- **`src/comprehensive_all_data_evaluator.py`** - Comprehensive testing
  - Cross-validation and model comparison
  - Multiple ML models testing (Random Forest, SVM, Gradient Boosting)

- **`src/run_accuracy_model.py`** - Model accuracy runner
  - Runs comprehensive evaluation pipeline
  - Generates performance reports

- **`src/enhanced_scoring.py`** - Alternative scoring approaches
  - Experimental scoring algorithms
  - Research on scoring improvements

- **`src/competency_assessor.py`** - Competency assessment
  - Analyzes specific competencies in answers
  - NLP-based competency detection

---

## 🗑️ Removed Files (Cleaned Up)

### Duplicates & Old Versions
- ❌ `main_old_backup.py` - Old backup of main.py
- ❌ `main_updated.py` - Duplicate/outdated version
- ❌ `interview_bot.py` - Old version (replaced by main.py)
- ❌ `question_selector.py` - Old selector (replaced by dataset_loader.py)
- ❌ `kaggle_dataset_loader.py` - Duplicate of dataset_loader.py

### One-Time Utility Scripts
- ❌ `check_unique_questions.py` - One-time utility to check question uniqueness
- ❌ `convert_kaggle_data.py` - One-time conversion script
- ❌ `create_enhanced_dataset.py` - One-time dataset creation
- ❌ `download_hr_analytics.py` - One-time download script
- ❌ `download_kaggle_data.py` - One-time download script
- ❌ `download_nltk.py` - One-time NLTK data download (now handled in tfidf_evaluator.py)

---

## 📂 Directory Structure

```
AI_Powered_Interview_Coach_Bot-_for_Job_Preparation/
│
├── src/                          # Source code
│   ├── main.py                   # ✅ Main application
│   ├── dataset_loader.py         # ✅ Dataset management
│   ├── tfidf_evaluator.py        # ✅ TF-IDF scoring (NEW)
│   ├── evaluator.py              # ✅ Behavioral evaluator
│   ├── logger.py                 # ✅ Session logging
│   ├── resources.py              # ✅ Tips & resources
│   ├── evaluate_model.py         # ✅ Model evaluation
│   │
│   └── [Research Files]          # 📊 Keep for analysis
│       ├── data_preprocessing_eda.py
│       ├── process_hr_data.py
│       ├── process_kaggle_data.py
│       ├── ai_vs_human_evaluator.py
│       ├── improved_ai_evaluator.py
│       ├── comprehensive_all_data_evaluator.py
│       ├── run_accuracy_model.py
│       ├── enhanced_scoring.py
│       └── competency_assessor.py
│
├── data/                         # Datasets
│   ├── questions.json
│   ├── questions_enhanced.json
│   └── kaggle_datasets/
│       └── deeplearning_questions.csv  # 111 technical questions
│
├── logs/                         # Session logs
│   └── session_log.txt
│
└── [Documentation]               # 📚 Markdown files
    ├── README.md
    ├── TF-IDF_SCORING_EXPLAINED.md
    ├── HOW_SMART_SYSTEM_WORKS.md
    ├── DATASET_MANAGEMENT_GUIDE.md
    ├── DATASET_STRUCTURE_EXPLAINED.md
    ├── FINAL_UPDATE_TWO_MODES.md
    ├── ROLE_FILTERING_UPDATE.md
    ├── AI_VS_HUMAN_GUIDE.md
    ├── CORE_TECHNOLOGIES_EXPLAINED.md
    ├── TECHNOLOGY_WORKFLOW.md
    └── PROJECT_SUMMARY.md
```

---

## 🚀 How to Run

### Main Interview Bot
```bash
cd src
python main.py
```

### Test TF-IDF Evaluator
```bash
cd src
python tfidf_evaluator.py
```

### Run Research/Analysis Scripts
```bash
cd src
python data_preprocessing_eda.py
python ai_vs_human_evaluator.py
python comprehensive_all_data_evaluator.py
```

---

## 📝 File Dependencies

### main.py imports:
- `dataset_loader.DatasetLoader`
- `evaluator.AnswerEvaluator`
- `tfidf_evaluator.TFIDFAnswerEvaluator`
- `resources.get_tip`
- `logger.log_response`
- `evaluate_model.InterviewBotEvaluator`

### No external dependencies on removed files ✅

---

## 🔄 Update History

**October 25, 2025:**
- ✅ Removed duplicate/old files (11 files)
- ✅ Kept all research and analysis files
- ✅ Verified main workflow still functional
- ✅ Added TF-IDF based scoring with NLTK
- ✅ Cleaned up codebase while preserving research work

---

## 💡 Notes

- All research files are kept for future reference and analysis
- Main workflow is streamlined to 7 core files
- TF-IDF evaluator now uses NLTK for better preprocessing
- Dataset loader handles all dataset management dynamically
- No hardcoded paths in the system
