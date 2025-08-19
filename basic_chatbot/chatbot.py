# basic_chatbot/chatbot.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure with your Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create a model instance (use gemini-1.5-flash or gemini-pro)
model = genai.GenerativeModel("gemini-1.5-flash")

def chat_with_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    print("Chatbot (Gemini) — type 'quit' to exit\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        print("Bot:", chat_with_gemini(user_input))




# # Basic GPT chatbot
# from openai import OpenAI
# import os

# client = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"))

# def chat_with_gpt(prompt):
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content

# if __name__ == "__main__":
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["quit", "exit"]:
#             break
#         print("Bot:", chat_with_gpt(user_input))
