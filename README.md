# 🩺 AI Medical Chatbot with RAG  

An **AI-powered medical assistant** that leverages **Retrieval-Augmented Generation (RAG)** to deliver accurate, context-aware responses to healthcare queries.  
The chatbot combines **document retrieval** with **Large Language Models (LLMs)** to ensure **factual, reliable, and source-backed answers**.  

🚀 Built with **LangChain, FAISS, Hugging Face embeddings, and Groq-hosted LLaMA-3 8B Instruct**, and deployed using **Streamlit**.  

---

## 📑 Table of Contents
1. [Overview](#-overview)
2. [Features](#-features)
3. [Tech Stack](#-tech-stack)
4. [System Architecture](#-system-architecture)
5. [Project Structure](#-project-structure)
6. [Installation](#-installation)
7. [Usage](#-usage)
8. [Environment Variables](#-environment-variables)
9. [Screenshots](#-screenshots)
10. [Demo](#-demo)
11. [Future Enhancements](#-future-enhancements)
12. [Contributing](#-contributing)
13. [License](#-license)

---

## 🔎 Overview
The **AI Medical Chatbot with RAG** is designed to assist users with **medical and healthcare-related queries** by retrieving context from **medical documents** and generating accurate responses.  
Unlike traditional chatbots, it integrates **vector databases + LLMs**, ensuring answers are **grounded in factual sources**.  

---

## 🚀 Features
- ✅ **Retrieval-Augmented Generation (RAG):** Combines retrieval with generation for better accuracy  
- ✅ **Medical Knowledge Integration:** Answers backed by stored medical context  
- ✅ **FAISS Vector Store:** Enables fast semantic search across medical documents  
- ✅ **Contextual Answers with Sources:** Avoids hallucinations and improves reliability  
- ✅ **Streamlit Interface:** Simple, interactive chatbot UI  
- ✅ **Scalable Deployment:** Compatible with Hugging Face Spaces / Streamlit Cloud  

---

## 🛠 Tech Stack
- **Programming Language:** Python  
- **Frameworks & Libraries:**  
  - LangChain  
  - FAISS  
  - Streamlit  
  - Hugging Face Transformers  
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`  
- **LLM Backend:** Groq-hosted `LLaMA-3 8B Instruct`  
- **Deployment:** Streamlit  

---

## 🏗 System Architecture
```mermaid
flowchart TD
    A[User Query] --> B[Streamlit UI]
    B --> C[LangChain RAG Pipeline]
    C --> D[FAISS Vector Store]
    D --> E[Relevant Context Retrieved]
    E --> F[Groq LLaMA-3 8B Instruct LLM]
    F --> G[Context-aware Response]
    G --> B
