---
# GenAI Starter Kit

A **14-day fast-track learning project** for Generative AI developers, built for **Python + FastAPI** enthusiasts.
This repo contains ready-to-run GenAI apps with **OpenAI API**, **LangChain**, **RAG**, and **Streamlit UIs**.
---

## Project Structure

```
genai-starter/
│
├── README.md                  # This guide
├── requirements.txt           # All dependencies
├── .env.example               # Example env vars for API keys
│
├── basic_chatbot/             # Basic GPT-powered chatbot
│   └── chatbot.py
│
├── rag_pdf_qa/                 # PDF-based question answering with RAG
│   ├── rag_pdf_qa.py
│   └── streamlit_rag.py        # Streamlit browser UI
│
├── fastapi_summarizer/         # Summarizer API
│   └── app.py
│
└── deploy/                     # Deployment scripts/templates
    ├── azure_webapp.yml
    └── streamlit_app.py
```

---

## Features

- **Basic Chatbot** using OpenAI API
- **RAG PDF QA Bot** (terminal + Streamlit browser UI)
- **FastAPI Summarizer API**
- Environment variable support with `.env`
- Ready-to-deploy scripts for **Azure** and **Streamlit Cloud**

---

## Setup

### Clone the repo

```bash
git clone https://github.com/lunatic-bot/genai-starter.git
cd genai-starter
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure API key

- Copy `.env.example` to `.env`
- Edit `.env`:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Get your key from [OpenAI API dashboard](https://platform.openai.com/).

---

## Running the Apps

### **Basic Chatbot**

```bash
python basic_chatbot/chatbot.py
```

---

### **RAG PDF QA Bot (Terminal)**

```bash
python rag_pdf_qa/rag_pdf_qa.py
```

> Make sure `your.pdf` is in the same folder or change the path in the script.

---

### **RAG PDF QA Bot (Streamlit UI)**

```bash
streamlit run rag_pdf_qa/streamlit_rag.py
```

- Opens in your browser
- Upload any PDF
- Ask questions directly in the UI

---

### **FastAPI Summarizer API**

```bash
uvicorn fastapi_summarizer.app:app --reload
```

- API available at: `http://127.0.0.1:8000/summarize`
- Example request (using `curl`):

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
-H "Content-Type: application/json" \
-d '{"text": "Your text to summarize here"}'
```

---

## Requirements

- Python **3.9+**
- OpenAI API key
- Internet connection (for API calls)

---

## Tech Stack

- **Python** — Core programming language
- **FastAPI** — API framework
- **Streamlit** — Web UI for GenAI apps
- **LangChain** — RAG + LLM integration
- **FAISS** — Vector search
- **pypdf** — PDF parsing
- **dotenv** — Environment variable management

---

## Next Steps

- Add **multi-turn memory** to chatbot
- Integrate **voice input/output**
- Deploy on **Azure** or **Streamlit Cloud**
- Experiment with **other LLMs** via Hugging Face

---

## License

This project is licensed under the MIT License — feel free to use and modify.

---
