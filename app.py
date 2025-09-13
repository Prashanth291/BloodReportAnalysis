import google.generativeai as genai

# Configure API key
genai.configure(api_key="AIzaSyD3Jwq0Qg4L5vDa4D3yl5yc78e3sdN0VvU")

# Upload your local file (PDF or image)
file = genai.upload_file("C:/Users/prash/OneDrive/Desktop/BloodReportAnalysis/inputs/image.png")

# Create the model
model = genai.GenerativeModel("gemini-2.5-pro")

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
        file
    ]
)

print(response.text)
