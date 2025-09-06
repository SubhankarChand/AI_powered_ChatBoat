import os
import requests
import json
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please create a .env file and add your API key.")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

# --- System Prompts for Different AI Modes ---
SYSTEM_PROMPTS = {
    "assistant": "You are a helpful AI assistant specializing in coding and algorithms for students. Provide clear explanations and format code snippets in Markdown.",
    "debugger": "You are an expert code debugger. A student will provide you with a piece of code that has a bug. Your task is to identify the bug, provide a corrected version of the code, and then give a clear, step-by-step explanation of what the original error was and how the fix works. Format the corrected code in a Markdown block.",
    "analyzer": "You are a computer science expert specializing in algorithm analysis. A student will provide you with a code snippet. Your task is to analyze it and determine its time and space complexity (Big O notation). Provide the complexities and a clear, simple explanation for your reasoning, as if explaining to a beginner.",
    "generator": "You are a creative mentor for student programmers. A student will list programming languages or concepts they know. Based on their input, your task is to generate 3 unique and achievable project ideas. For each idea, provide a brief description and list the key concepts they will use."
}

# --- Routes ---
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles requests from the frontend, constructs the correct prompt, and calls the Gemini API."""
    user_data = request.get_json()
    if not user_data or 'prompt' not in user_data or 'mode' not in user_data:
        return jsonify({"error": "Invalid input. 'prompt' and 'mode' keys are required."}), 400

    prompt = user_data['prompt']
    mode = user_data['mode']

    # Get the appropriate system prompt for the selected mode
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS['assistant'])
    
    # Combine the system instruction with the user's prompt
    full_prompt = f"{system_prompt}\n\nHere is the user's input:\n\n{prompt}"

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}]
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        gemini_response = response.json()
        
        generated_text = gemini_response['candidates'][0]['content']['parts'][0]['text']
        return jsonify({"generated_text": generated_text})

    except requests.exceptions.RequestException as e:
        error_details = str(e)
        try:
            error_details = response.json()
        except Exception: pass
        return jsonify({"error": "Failed to get a response from the Gemini API.", "details": error_details}), 500
    except (KeyError, IndexError) as e:
        return jsonify({"error": "Could not parse the response from the Gemini API.", "details": str(gemini_response)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

