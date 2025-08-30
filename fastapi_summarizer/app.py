# FastAPI summarizer API using Gemini
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# Serve /static and index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse("static/index.html")

class InputText(BaseModel):
    text: str

# Create Gemini model instance
model = genai.GenerativeModel("gemini-1.5-flash")

@app.post("/summarize")
def summarize(data: InputText):
    prompt = f"Summarize the following text in 3 concise bullet points:\n\n{data.text}"
    response = model.generate_content(prompt)
    return {"summary": response.text}

