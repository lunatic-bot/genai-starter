# FastAPI summarizer API using Gemini
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

class InputText(BaseModel):
    text: str

# Create Gemini model instance
model = genai.GenerativeModel("gemini-1.5-flash")

@app.post("/summarize")
def summarize(data: InputText):
    # Build the prompt
    prompt = f"Summarize the following text in 3 concise bullet points:\n\n{data.text}"
    
    # Call Gemini
    response = model.generate_content(prompt)
    
    return {"summary": response.text}


