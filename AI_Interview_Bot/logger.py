import os
from datetime import datetime

def log_response(question, user_answer, score, feedback):
    """
    Log interview responses to a file.
    
    Args:
        question (str): The question asked
        user_answer (str): The user's response
        score (float): Evaluation score
        feedback (str): Feedback provided
    """
    try:
        # Ensure logs directory exists
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        log_file = os.path.join(logs_dir, "session_log.txt")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("---- New Response ----\n")
            f.write(f"Timestamp : {datetime.now()}\n")
            f.write(f"Question  : {question}\n")
            f.write(f"Answer    : {user_answer}\n")
            f.write(f"Score     : {round(float(score), 2)}\n")
            f.write(f"Feedback  : {feedback}\n\n")
    
    except Exception as e:
        print(f"Warning: Failed to log response: {str(e)}")
