# FastAPI summarizer API
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/summarize")
def summarize(data: InputText):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize in 3 bullet points."},
            {"role": "user", "content": data.text}
        ]
    )
    return {"summary": response.choices[0].message.content}
