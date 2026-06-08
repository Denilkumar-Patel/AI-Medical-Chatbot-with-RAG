import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
DB_FAISS_PATH = Path(
    os.getenv("DB_FAISS_PATH", str(BASE_DIR / "vectorstore" / "db_faiss"))
).expanduser()
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
GROQ_MODEL = os.getenv(
    "GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"
)

load_dotenv(BASE_DIR / ".env")


CUSTOM_PROMPT_TEMPLATE = """
You are an educational medical information assistant.
Use only the context below to answer the user's question.
If the answer is not present in the context, say you do not know based on the available sources.
Do not invent diagnoses, treatments, dosages, or emergency guidance.
Always remind users to consult a qualified healthcare professional for medical decisions.

Context:
{context}

Question:
{question}

Answer directly, then include a short "Medical safety note".
"""


def validate_runtime_config() -> list[str]:
    issues = []
    if not os.getenv("GROQ_API_KEY"):
        issues.append("GROQ_API_KEY is not configured.")
    if not (DB_FAISS_PATH / "index.faiss").exists():
        issues.append(f"Missing FAISS index: {DB_FAISS_PATH / 'index.faiss'}")
    if not (DB_FAISS_PATH / "index.pkl").exists():
        issues.append(f"Missing FAISS metadata: {DB_FAISS_PATH / 'index.pkl'}")
    return issues


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_vectorstore(path: Path = DB_FAISS_PATH) -> FAISS:
    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Vector store not found: {resolved_path}")

    # FAISS stores LangChain metadata in a pickle file. Only load indexes created by this app.
    return FAISS.load_local(
        str(resolved_path),
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )


def build_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required to call the Groq model.")

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        groq_api_key=api_key,
    )


def build_qa_chain(vectorstore: FAISS, k: int = 4) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=build_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": build_prompt()},
    )


def format_source_documents(source_documents) -> str:
    if not source_documents:
        return "No source documents returned."

    formatted = []
    seen = set()
    for document in source_documents:
        metadata = document.metadata or {}
        source = metadata.get("source", "Unknown source")
        page = metadata.get("page")
        label = f"{Path(source).name}"
        if page is not None:
            label += f", page {int(page) + 1}"
        if label in seen:
            continue
        seen.add(label)
        formatted.append(f"- {label}")

    return "\n".join(formatted) if formatted else "No source metadata returned."
