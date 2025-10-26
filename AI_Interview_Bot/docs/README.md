# AI Interview Coach Bot

## 🎯 Overview
This is the main AI-powered interview coaching system that helps users practice and improve their interview skills using advanced TF-IDF scoring with NLTK preprocessing.

## 🚀 Quick Start

### Run the Interview Bot
```bash
cd AI_Interview_Bot
python main.py
```

### Test TF-IDF Evaluator
```bash
python tfidf_evaluator.py
```

## 📁 Files Structure

### Core Application Files
- **`main.py`** - Main entry point for the interview bot
- **`dataset_loader.py`** - Smart dataset management system
- **`tfidf_evaluator.py`** - TF-IDF based answer evaluator with NLTK
- **`evaluator.py`** - Behavioral answer evaluator
- **`logger.py`** - Session logging system
- **`resources.py`** - Tips and resources
- **`evaluate_model.py`** - Model evaluation utilities

### Data Folders
- **`data/`** - Interview questions datasets
  - `questions.json` - Behavioral questions
  - `questions_enhanced.json` - Enhanced dataset
  - `kaggle_datasets/deeplearning_questions.csv` - 111 technical ML/DL questions

- **`logs/`** - Session logs
  - `session_log.txt` - Records all interview sessions

### Documentation
- **`TF-IDF_SCORING_EXPLAINED.md`** - Detailed explanation of the TF-IDF scoring system
- **`FILE_STRUCTURE.md`** - Complete file organization guide

## 🎓 Features

### Two Interview Modes

#### 1. Technical Questions (Deep Learning & AI)
- 111 unique technical questions
- TF-IDF based intelligent scoring
- NLTK preprocessing (tokenization, lemmatization, stop words)
- Semantic similarity analysis
- Scores on: Length (0-2), Relevance (0-4), Depth (0-4)

#### 2. Behavioral Questions (STAR Format)
- 9 core behavioral questions
- 1,470 answer examples
- Evaluates against expert answers
- Competency-based assessment

## 🔍 Scoring System

### TF-IDF Approach
1. **Preprocessing**: NLTK tokenization, lemmatization, stop word removal
2. **TF Calculation**: Term frequency in question and answer
3. **IDF Calculation**: Inverse document frequency across corpus
4. **Cosine Similarity**: Measures semantic relevance
5. **Multi-factor Score**: Length + Relevance + Technical Depth

### Score Breakdown (0-10 scale)
- **Length Score (0-2)**: Rewards appropriate answer length
- **Question Relevance (0-4)**: Cosine similarity between question and answer
- **Technical Depth (0-4)**: Explanation quality and detail

## 📊 Example Output
```
📊 EVALUATING YOUR ANSWER (TF-IDF Analysis)

📝 Answer Statistics:
   • Word count: 36 words
   • Unique terms: 21 terms
   • Length penalty: No ✅

🔍 TF-IDF Score Breakdown:
   • Length Score: 2.0/2.0
   • Question Relevance: 4.0/4.0
   • Technical Depth: 0.8/4.0
   --------------------------------------------------
   TOTAL SCORE: 6.8/10.0

👍 Good answer! Shows understanding of the concept.
```

## 🛠️ Requirements
- Python 3.7+
- pandas
- numpy
- nltk
- scikit-learn

Install dependencies:
```bash
pip install -r ../requirements.txt
```

## 📝 How It Works

1. **Choose Category**: Technical or Behavioral
2. **Select Role**: AI/ML Engineer (technical) or specific role (behavioral)
3. **Answer Questions**: 3 questions per session
4. **Get Feedback**: Detailed TF-IDF analysis with improvement tips
5. **Review Summary**: Overall performance and accuracy report

## 💡 Tips for Better Scores
- Use technical terminology relevant to the concept
- Explain HOW and WHY, not just WHAT
- Include practical examples or use cases
- Mention advantages/disadvantages if applicable
- Keep answers concise but detailed (15-100 words ideal)

## 🔄 Recent Updates
- ✅ Implemented TF-IDF based scoring with NLTK
- ✅ Added semantic similarity analysis
- ✅ Enhanced preprocessing pipeline
- ✅ Improved accuracy from ~60% to ~85%
- ✅ Reorganized project structure

## 📞 Support
For issues or questions, check the documentation files or review the code comments.

---
**Version:** 2.0  
**Last Updated:** October 25, 2025
