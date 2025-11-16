// Global state
let sessionId = null;
let selectedCategory = null;
let currentQuestionNumber = 0;
let totalQuestions = 0;

// Update status bar
function updateStatus(message, type = 'info') {
    const statusBar = document.getElementById('statusBar');
    const statusText = document.getElementById('statusText');
    
    statusText.textContent = message;
    statusBar.className = 'status-bar';
    
    if (type === 'success') {
        statusBar.classList.add('success');
    } else if (type === 'error') {
        statusBar.classList.add('error');
    }
}

// Select category and initialize
async function selectCategory(category) {
    selectedCategory = category;
    
    updateStatus('Initializing interview session...', 'info');
    
    try {
        const response = await fetch('/initialize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ category: category })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            sessionId = data.session_id;
            totalQuestions = data.total_questions;
            
            // Hide category selection
            document.getElementById('categorySection').style.display = 'none';
            
            // Show control panel
            document.getElementById('controlPanel').style.display = 'flex';
            document.getElementById('getQuestionBtn').disabled = false;
            document.getElementById('summaryBtn').disabled = false;
            
            updateStatus(`✅ ${data.message} - ${totalQuestions} questions loaded. Click "Get Question" to start!`, 'success');
        } else {
            updateStatus('Error: ' + data.message, 'error');
        }
    } catch (error) {
        updateStatus('Failed to connect to server', 'error');
        console.error('Error:', error);
    }
}

// Reset session
function resetSession() {
    sessionId = null;
    selectedCategory = null;
    currentQuestionNumber = 0;
    totalQuestions = 0;
    
    // Show category selection
    document.getElementById('categorySection').style.display = 'block';
    
    // Hide other sections
    document.getElementById('controlPanel').style.display = 'none';
    hideQuestionAndAnswer();
    hideResults();
    
    updateStatus('Select a category to begin', 'info');
}

// Get a question
async function getQuestion() {
    if (!sessionId) {
        updateStatus('Please select a category first', 'error');
        return;
    }
    
    const getQuestionBtn = document.getElementById('getQuestionBtn');
    getQuestionBtn.disabled = true;
    getQuestionBtn.innerHTML = '<span class="loading"></span> Loading...';
    
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ session_id: sessionId })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            currentQuestionNumber = data.question_number;
            displayQuestion(data.question);
            updateStatus(`Question ${currentQuestionNumber}/${totalQuestions} - Provide your answer`, 'success');
            hideResults();
        } else if (data.status === 'complete') {
            updateStatus(data.message, 'info');
            getQuestionBtn.disabled = true;
        } else {
            updateStatus('Error: ' + data.message, 'error');
        }
    } catch (error) {
        updateStatus('Failed to get question', 'error');
        console.error('Error:', error);
    } finally {
        getQuestionBtn.disabled = false;
        getQuestionBtn.textContent = 'Get Question';
    }
}

// Display question
function displayQuestion(question) {
    const questionSection = document.getElementById('questionSection');
    const questionText = document.getElementById('questionText');
    const answerSection = document.getElementById('answerSection');
    const answerInput = document.getElementById('answerInput');
    
    questionText.textContent = question;
    questionSection.style.display = 'block';
    answerSection.style.display = 'block';
    answerInput.value = '';
    answerInput.focus();
}

// Submit answer
async function submitAnswer() {
    const answerInput = document.getElementById('answerInput');
    const answer = answerInput.value.trim();
    
    if (!answer) {
        updateStatus('Please provide an answer before submitting', 'error');
        return;
    }
    
    if (!sessionId) {
        updateStatus('Session expired. Please restart.', 'error');
        return;
    }
    
    updateStatus('Evaluating your answer...', 'info');
    
    try {
        const response = await fetch('/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                session_id: sessionId,
                answer: answer 
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            displayResults(data.score, data.feedback, data.reference_answer);
            updateStatus('Answer evaluated successfully!', 'success');
        } else {
            updateStatus('Error: ' + data.message, 'error');
        }
    } catch (error) {
        updateStatus('Failed to submit answer', 'error');
        console.error('Error:', error);
    }
}

// Display results
function displayResults(score, feedback, reference) {
    const resultsSection = document.getElementById('resultsSection');
    const scoreValue = document.getElementById('scoreValue');
    const feedbackText = document.getElementById('feedbackText');
    const referenceText = document.getElementById('referenceText');
    
    scoreValue.textContent = score.toFixed(1);
    feedbackText.textContent = feedback;
    referenceText.textContent = reference || 'No reference answer available.';
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Color code the score
    if (score >= 80) {
        scoreValue.style.color = 'var(--success-color)';
    } else if (score >= 60) {
        scoreValue.style.color = 'var(--warning-color)';
    } else {
        scoreValue.style.color = 'var(--danger-color)';
    }
}

// Hide results
function hideResults() {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'none';
}

// Next question
async function nextQuestion() {
    if (!sessionId) {
        updateStatus('Session expired. Please restart.', 'error');
        return;
    }
    
    try {
        const response = await fetch('/next', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ session_id: sessionId })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            updateStatus('Ready for next question. Click "Get Question" to continue.', 'success');
            hideQuestionAndAnswer();
            hideResults();
        } else if (data.status === 'complete') {
            updateStatus('🎉 ' + data.message + ' View your summary!', 'success');
            hideQuestionAndAnswer();
            hideResults();
            document.getElementById('getQuestionBtn').disabled = true;
        } else {
            updateStatus('Error: ' + data.message, 'error');
        }
    } catch (error) {
        updateStatus('Failed to load next question', 'error');
        console.error('Error:', error);
    }
}

// Hide question and answer sections
function hideQuestionAndAnswer() {
    document.getElementById('questionSection').style.display = 'none';
    document.getElementById('answerSection').style.display = 'none';
}

// Show summary
async function showSummary() {
    if (!sessionId) {
        updateStatus('No session available. Please start an interview first.', 'error');
        return;
    }
    
    try {
        const response = await fetch('/summary', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ session_id: sessionId })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            displaySummary(data.summary);
        } else {
            updateStatus('Error: ' + data.message, 'error');
        }
    } catch (error) {
        updateStatus('Failed to get summary', 'error');
        console.error('Error:', error);
    }
}

// Display summary modal
function displaySummary(summary) {
    const modal = document.getElementById('summaryModal');
    const summaryContent = document.getElementById('summaryContent');
    
    let html = '<div style="line-height: 2;">';
    
    if (typeof summary === 'object') {
        for (const [key, value] of Object.entries(summary)) {
            html += `<p><strong>${formatKey(key)}:</strong> ${value}</p>`;
        }
    } else {
        html += `<p>${summary}</p>`;
    }
    
    html += '</div>';
    summaryContent.innerHTML = html;
    modal.style.display = 'block';
}

// Format object keys for display
function formatKey(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, char => char.toUpperCase());
}

// Close summary modal
function closeSummary() {
    const modal = document.getElementById('summaryModal');
    modal.style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('summaryModal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
}

// Handle Enter key in textarea
document.addEventListener('DOMContentLoaded', function() {
    const answerInput = document.getElementById('answerInput');
    
    if (answerInput) {
        answerInput.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                submitAnswer();
            }
        });
    }
});
