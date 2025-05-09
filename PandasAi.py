from flask import Flask, request, render_template, jsonify
import pandas as pd
import os
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
from docx import Document

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API Key (ensure it's set in your environment for security)
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your actual API key

# Global variables
sdf = None
conversation_history = []  # List to store conversation history

@app.route('/', methods=['GET'])
def home():
    return render_template('front.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global sdf, conversation_history

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        try:
            file_ext = file.filename.split('.')[-1].lower()
            if file_ext == 'csv':
                df = pd.read_csv(file)
            elif file_ext in ['txt', 'log']:
                df = pd.DataFrame({'text': file.readlines()})
            elif file_ext == 'docx':
                doc = Document(file)
                df = pd.DataFrame({'text': [para.text for para in doc.paragraphs]})
            else:
                return jsonify({'error': 'Unsupported file format'}), 400

            # Initialize OpenAI LLM and SmartDataframe
            llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            sdf = SmartDataframe(df, config={"llm": llm})
            conversation_history = []  # Reset history on new file upload

            return jsonify({'message': 'File uploaded and SmartDataframe initialized successfully'}), 200
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Please upload a valid file'}), 400

@app.route('/query', methods=['POST'])
def query_data():
    global sdf, conversation_history

    if sdf is None:
        return jsonify({'error': 'Please upload a file first'}), 400

    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({'error': 'Please provide a question'}), 400

    try:
        # Build context from conversation history
        full_prompt = "Here is the conversation history for context:\n"
        for entry in conversation_history:
            full_prompt += f"Q: {entry['prompt']}\nA: {entry['answer']}\n"
        full_prompt += f"New question: {prompt}"

        # Use PandasAI to process the query with context
        response = sdf.chat(full_prompt)

        # Append to conversation history
        conversation_history.append({'prompt': prompt, 'answer': response})

        # Return the full history
        return jsonify({'history': conversation_history}), 200
    except Exception as e:
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)