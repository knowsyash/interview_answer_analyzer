# AI Interview Coach Bot - Web Interface

## 🌐 Professional GUI for Interview Practice

This web interface provides a clean, professional way to interact with the AI Interview Coach Bot through your browser.

## 📋 Features

- **Modern UI**: Clean, responsive design with smooth animations
- **Multiple Categories**: Choose from Web Development, Python, Java, JavaScript, Database, or Behavioral interviews
- **Real-time Evaluation**: Get instant feedback on your answers with detailed scoring
- **Session Tracking**: Track your progress across multiple questions
- **Summary Reports**: View comprehensive session summaries with performance metrics
- **Reference Answers**: Compare your answers with expert reference answers

## 🚀 Quick Start

### 1. Install Dependencies

First, make sure you have Flask installed:

```powershell
pip install -r requirements_web.txt
```

Or install manually:

```powershell
pip install Flask
```

### 2. Run the Application

Start the web server:

```powershell
python app.py
```

The application will start on `http://localhost:5000`

### 3. Open Your Browser

Navigate to: **http://localhost:5000**

## 📖 How to Use

1. **Select Category**: Choose your interview domain from the available categories
2. **Get Question**: Click "Get Question" to receive an interview question
3. **Provide Answer**: Type your answer in the text area
4. **Submit**: Click "Submit Answer" to get your evaluation
5. **Review Results**: See your score, feedback, and reference answer
6. **Continue**: Click "Next Question" to move to the next question
7. **View Summary**: Check your overall performance anytime

## 🎨 Features Overview

### Category Selection
- Web Development (HTML, CSS, JavaScript, React, Node.js)
- Python Programming (Django, Flask, Pandas, NumPy)
- Java Development (Spring, Hibernate, Maven, Android)
- JavaScript/Node.js (React, Angular, TypeScript)
- Database/SQL (MySQL, PostgreSQL, MongoDB)
- Behavioral (STAR Format, Leadership, Communication)

### Evaluation System
- TF-IDF based similarity scoring
- Reference answer comparison
- Detailed feedback with improvement tips
- Score breakdown and metrics

### User Interface
- Responsive design (works on desktop and mobile)
- Color-coded scores (Green: Excellent, Yellow: Good, Red: Needs Improvement)
- Smooth animations and transitions
- Intuitive navigation

## 🔧 Technical Details

### File Structure
```
AI_Interview_Bot/
├── app.py                    # Flask application
├── templates/
│   └── index.html           # Main HTML template
├── static/
│   ├── style.css            # Stylesheet
│   └── script.js            # JavaScript functions
├── data/                     # Question datasets
├── evaluator.py             # Evaluation logic
├── tfidf_evaluator.py       # TF-IDF scoring
└── requirements_web.txt     # Web dependencies
```

### API Endpoints

- `POST /initialize` - Initialize interview session with category
- `POST /ask` - Get next question
- `POST /submit` - Submit answer for evaluation
- `POST /next` - Move to next question
- `POST /summary` - Get session summary

## 💡 Tips for Best Results

1. **Be Detailed**: Provide comprehensive answers (aim for 20-50 words minimum)
2. **Use Technical Terms**: Include relevant terminology from your field
3. **Explain Your Thinking**: Don't just state facts, explain the "why" and "how"
4. **Review Feedback**: Read the feedback carefully to improve on next answers
5. **Compare with References**: Study the reference answers to understand what makes a good response

## 🎯 Keyboard Shortcuts

- `Ctrl + Enter` in answer textarea - Submit your answer

## 🛠️ Troubleshooting

### Server won't start
- Make sure Flask is installed: `pip install Flask`
- Check if port 5000 is available
- Try a different port: Edit `app.py` and change `port=5000` to another port

### Questions not loading
- Verify the `data/` folder exists with CSV files
- Check console for error messages
- Ensure all Python dependencies are installed

### Evaluation errors
- Make sure all evaluator files are present
- Check that required models (joblib files) are in the correct location

## 📝 Notes

- The web interface uses the same evaluation logic as the command-line version
- Session data is stored in memory (restarting the server clears all sessions)
- Each browser session gets a unique session ID
- You can run multiple interviews in different browser tabs

## 🔄 Resetting

To start a new interview:
1. Click "Change Category" button
2. Select a new category
3. Begin your new session

## 📊 Performance Metrics

Scores are calculated based on:
- Answer length and completeness
- Keyword and terminology usage
- Similarity to reference answers
- Technical accuracy
- Clarity of explanation

Scoring scale:
- **8-10**: Excellent answer
- **6-7.9**: Good answer
- **4-5.9**: Needs improvement
- **Below 4**: Insufficient answer

---

**Enjoy your interview preparation! 🚀**
