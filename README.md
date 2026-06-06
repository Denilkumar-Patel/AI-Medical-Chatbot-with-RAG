# AI Medical Chatbot with RAG

AI Medical Chatbot with RAG is a Streamlit application that answers medical education questions using a FAISS vector store built from bundled medical reference material. The app retrieves relevant context, sends it to a Groq-hosted LLM through LangChain, and returns a grounded answer with source pages.

This project is for educational information only. It is not a diagnostic tool and must not be used as a substitute for professional medical advice, emergency care, or clinical judgment.

## Features

- Retrieval-augmented generation with FAISS and LangChain
- Hugging Face sentence-transformer embeddings
- Groq-hosted chat model for response generation
- Source page display instead of dumping raw document objects
- Stable path handling for local runs, Streamlit Cloud, and Docker
- Runtime checks for missing API keys and FAISS index files
- Dockerfile, Streamlit config, and CI compile check

## Project Structure

```text
.
├── Dockerfile
├── README.md
├── medical-chatbot-main/
│   ├── medibot.py
│   ├── rag_core.py
│   ├── create_memory_for_llm.py
│   ├── connect_memory_with_llm.py
│   ├── requirements.txt
│   ├── data/
│   └── vectorstore/db_faiss/
```

## Local Setup

```bash
cd medical-chatbot-main
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy ..\.env.example .env
```

Set your Groq key in `.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

If the FAISS index is missing or you change the PDFs, rebuild it:

```bash
python create_memory_for_llm.py
```

Run the app:

```bash
streamlit run medibot.py
```

Run a CLI query:

```bash
python connect_memory_with_llm.py
```

## Deployment

### Streamlit Community Cloud

1. Set the main file path to `medical-chatbot-main/medibot.py`.
2. Add `GROQ_API_KEY` in app secrets.
3. Keep `medical-chatbot-main/vectorstore/db_faiss/index.faiss` and `index.pkl` available, or rebuild the index before deployment.

### Docker

```bash
docker build -t medical-rag-chatbot .
docker run -e GROQ_API_KEY=your_groq_api_key -p 8501:8501 medical-rag-chatbot
```

## Runtime Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | required | Groq model access |
| `GROQ_MODEL` | `meta-llama/llama-4-maverick-17b-128e-instruct` | Chat model |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `DB_FAISS_PATH` | `medical-chatbot-main/vectorstore/db_faiss` | Vector store path |

## Safety Notes

The model is constrained to answer from retrieved context, but generated medical text can still be incomplete or wrong. The UI includes a safety reminder and should be positioned as an educational assistant, not a medical decision system.
