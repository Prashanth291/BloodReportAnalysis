import google.generativeai as genai
from flask import Flask, request, render_template, jsonify
import os

app = Flask(__name__)

# Configure API key
# It's recommended to use environment variables for API keys
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "AIzaSyD3Jwq0Qg4L5vDa4D3yl5yc78e3sdN0VvU"))

@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/analyze')
def analyze_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            # Read file bytes and get mimetype
            file_bytes = file.read()
            mime_type = file.mimetype

            # Create the model
            model = genai.GenerativeModel("gemini-2.5-pro") # Switched to gemini-pro-vision for a more generous free tier

            # Prepare the file for the model
            uploaded_file = {
                'mime_type': mime_type,
                'data': file_bytes
            }

            # Send prompt + file
            response = model.generate_content(
                [
                    """You are a medical report parser. 
                    Extract laboratory values from the provided medical report and return them in strict JSON format. 
                    Do not include explanations or extra text. 

                    Schema:
                    {
                      "Hemoglobin": "<value with units>",
                      "WBC": "<value with units>",
                      "RBC": "<value with units>",
                      "Platelets": "<value with units>",
                      "GlucoseFasting": "<value with units>",
                      "Cholesterol": "<value with units>",
                      "OtherParameters": {
                        "<ParameterName>": "<Value with Units>"
                      }
                    }
                    If a parameter is missing, return null.
                    """,
                    uploaded_file
                ]
            )
            return jsonify({"data": response.text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
