# AI Interview Answer Analyzer

An intelligent web application that evaluates interview answers using machine learning algorithms, providing instant feedback and scoring based on reference answers and trained models.

## 🌐 Live Demo

**[Try it live!](https://interview-answer-analyzer.onrender.com/)**

## 📋 Overview

The AI Interview Answer Analyzer is a Flask-based web application designed to help interview candidates practice and receive feedback on their technical interview answers. The system uses a combination of TF-IDF (Term Frequency-Inverse Document Frequency) and Random Forest machine learning models to evaluate responses and provide scores along with detailed feedback.

## ✨ Features

- **Real-time Answer Evaluation**: Get instant feedback on your interview answers
- **Multiple Evaluation Methods**: 
  - TF-IDF-based semantic similarity scoring
  - Random Forest ML model trained on real interview data
- **Interactive Web Interface**: User-friendly UI for seamless interaction
- **Question Bank**: Comprehensive collection of web development interview questions
- **Score Tracking**: Monitor your progress across multiple questions
- **Detailed Feedback**: Receive constructive feedback on your answers

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: 
  - scikit-learn (Random Forest, TF-IDF Vectorizer)
  - NLTK (Natural Language Processing)
  - pandas (Data Processing)
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render
- **Model Storage**: joblib for model persistence

## 📁 Project Structure

```
interview_answer_analyzer/
├── AI_Interview_Bot/
│   ├── app.py                          # Main Flask application
│   ├── random_forest_evaluator.py     # Random Forest model evaluator
│   ├── tfidf_evaluator.py             # TF-IDF based evaluator
│   ├── reference_answer_loader.py     # Reference answer handler
│   ├── dataset_loader.py              # Dataset loading utilities
│   ├── logger.py                      # Logging configuration
│   ├── resources.py                   # Resource management
│   ├── real_dataset_score/            # Training data and models
│   │   ├── random_forest_model.joblib # Trained ML model
│   │   ├── combined_training_data.csv # Training dataset
│   │   └── webdev_interview_qa.csv    # Interview Q&A pairs
│   ├── static/                        # CSS and JavaScript files
│   │   ├── style.css
│   │   └── script.js
│   └── templates/                     # HTML templates
│       └── index.html
├── Research_Analysis/
│   └── Optimized_Model_Training.ipynb # Model training notebook
├── requirements.txt                    # Python dependencies
├── render.yaml                        # Render deployment config
└── build.sh                           # Build script
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd interview_answer_analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   cd AI_Interview_Bot
   python app.py
   ```

4. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 💡 Usage

1. **Start the Application**: Visit the live demo or run locally
2. **Answer Questions**: Read the interview question and type your answer
3. **Submit & Evaluate**: Click submit to get your answer evaluated
4. **Review Feedback**: Check your score and read the detailed feedback
5. **Continue Practice**: Move to the next question to continue improving

## 🎯 Scoring System

The application uses a dual-evaluation approach:

- **TF-IDF Similarity**: Measures semantic similarity between your answer and reference answers
- **Random Forest Model**: ML model trained on real interview data to predict answer quality
- **Final Score**: Combined score providing comprehensive evaluation (0-100 scale)

## 🧠 Model Training

The Random Forest model is trained on:
- Real interview answer datasets
- Stack Overflow Q&A pairs
- Web development interview questions and answers
- Combined training data with quality scores

Training notebook available in `Research_Analysis/Optimized_Model_Training.ipynb`

## 🌐 Deployment

The application is deployed on Render. Deployment configuration:
- **Platform**: Render
- **Type**: Web Service
- **Build Command**: `./build.sh`
- **Start Command**: Defined in `render.yaml`

## 📝 Dependencies

Main dependencies include:
- Flask
- scikit-learn
- pandas
- nltk
- joblib

For a complete list, see [requirements.txt](requirements.txt)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is available for educational and personal use.

## 🔗 Links

- **Live Application**: https://interview-answer-analyzer.onrender.com/
- **Repository**: [Add your repository URL here]

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

Made with ❤️ for interview preparation and practice
