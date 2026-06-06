from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

from rag_core import DATA_PATH, DB_FAISS_PATH, get_embedding_model


def load_pdf_files(data_path=DATA_PATH):
    loader = DirectoryLoader(str(data_path), glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)


def main() -> None:
    documents = load_pdf_files()
    if not documents:
        raise RuntimeError(f"No PDF files found in {DATA_PATH}")

    text_chunks = create_chunks(documents)
    db = FAISS.from_documents(text_chunks, get_embedding_model())
    DB_FAISS_PATH.mkdir(parents=True, exist_ok=True)
    db.save_local(str(DB_FAISS_PATH))
    print(f"Saved FAISS vector store to {DB_FAISS_PATH}")


if __name__ == "__main__":
    main()
