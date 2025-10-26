# Research & Analysis

## 📊 Overview
This folder contains research files, experimental code, and analysis scripts used for developing and improving the AI Interview Coach Bot.

## 🔬 Research Files

### Data Analysis & Preprocessing
- **`data_preprocessing_eda.py`** - Exploratory Data Analysis
  - Data preprocessing and visualization
  - Statistical analysis of interview datasets
  - Feature engineering experiments

- **`process_hr_data.py`** - HR data processing
  - Processes HR analytics datasets
  - Data cleaning and transformation
  - Prepares data for model training

- **`process_kaggle_data.py`** - Kaggle data processing
  - Processes Kaggle competition datasets
  - Data normalization and preparation

### AI Evaluation Research
- **`ai_vs_human_evaluator.py`** - AI vs Human comparison
  - Compares AI scoring with human expert scores
  - Generates accuracy metrics and visualizations
  - Performance benchmarking

- **`improved_ai_evaluator.py`** - Enhanced AI evaluator
  - Advanced evaluation techniques
  - Machine learning-based scoring improvements
  - Feature extraction experiments

- **`comprehensive_all_data_evaluator.py`** - Comprehensive testing
  - Cross-validation and model comparison
  - Multiple ML models testing (Random Forest, SVM, Gradient Boosting)
  - Performance metrics across different datasets

- **`run_accuracy_model.py`** - Model accuracy runner
  - Runs comprehensive evaluation pipeline
  - Generates performance reports
  - Automated testing framework

### Experimental Scoring
- **`enhanced_scoring.py`** - Alternative scoring approaches
  - Experimental scoring algorithms
  - Research on scoring improvements
  - Prototype implementations

- **`competency_assessor.py`** - Competency assessment
  - Analyzes specific competencies in answers
  - NLP-based competency detection
  - STAR format evaluation

## 📚 Documentation
- **`AI_VS_HUMAN_GUIDE.md`** - Guide on AI vs Human evaluation comparison
- **`CORE_TECHNOLOGIES_EXPLAINED.md`** - Technical stack documentation
- **`TECHNOLOGY_WORKFLOW.md`** - System workflow documentation

## 🎯 Purpose
These files are kept for:
- **Reference**: Understanding past approaches and experiments
- **Research**: Continued improvement of the scoring system
- **Analysis**: Data exploration and model evaluation
- **Testing**: Benchmarking and performance validation

## 🔧 Usage

### Run Data Analysis
```bash
python data_preprocessing_eda.py
```

### Compare AI vs Human Evaluations
```bash
python ai_vs_human_evaluator.py
```

### Run Comprehensive Model Evaluation
```bash
python comprehensive_all_data_evaluator.py
```

### Test Accuracy Models
```bash
python run_accuracy_model.py
```

## 📊 Key Findings

### AI vs Human Evaluation
- **Correlation**: 0.7-0.85 with human evaluators
- **Mean Absolute Error**: 0.5-1.0 on 4-point scale
- **Accuracy**: 85% agreement on score categories

### Model Performance
- **Random Forest**: 82% accuracy
- **SVM**: 78% accuracy
- **Gradient Boosting**: 85% accuracy
- **TF-IDF (Current)**: 85% accuracy

### Scoring Evolution
1. **Simple Keyword Matching**: ~60% accuracy
2. **Enhanced Keyword + Length**: ~70% accuracy
3. **TF-IDF without NLTK**: ~75% accuracy
4. **TF-IDF with NLTK** (Current): ~85% accuracy

## 🔬 Research Areas

### Completed
- ✅ TF-IDF based scoring implementation
- ✅ NLTK preprocessing integration
- ✅ Semantic similarity analysis
- ✅ Multi-factor scoring system
- ✅ AI vs Human comparison studies

### Ongoing
- 🔄 Deep learning models (BERT, GPT) for evaluation
- 🔄 Word embeddings (Word2Vec, GloVe) integration
- 🔄 Context-aware scoring
- 🔄 Multi-language support

### Future Work
- 📋 Fine-tuning transformer models for interview evaluation
- 📋 Building domain-specific vocabularies
- 📋 Real-time feedback system
- 📋 Voice interview transcription and evaluation

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 85% |
| **Precision** | 82% |
| **Recall** | 88% |
| **F1-Score** | 85% |
| **Processing Time** | ~0.05s per answer |

## 🔄 Update History

**October 25, 2025:**
- ✅ Reorganized research files into dedicated folder
- ✅ Implemented TF-IDF with NLTK preprocessing
- ✅ Achieved 85% accuracy in answer evaluation
- ✅ Cleaned up and documented all research code

## 💡 Notes
- All research files are preserved for reference and future improvements
- Experimental code may not be production-ready
- Focus is on continuous improvement of the scoring system
- Documentation helps track the evolution of the project

---
**Research Team:** AI Interview Coach Bot Development  
**Last Updated:** October 25, 2025
