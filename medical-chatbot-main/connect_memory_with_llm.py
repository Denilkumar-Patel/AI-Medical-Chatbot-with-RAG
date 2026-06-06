from rag_core import build_qa_chain, format_source_documents, load_vectorstore


def main() -> None:
    qa_chain = build_qa_chain(load_vectorstore())
    user_query = input("Write query here: ").strip()
    if not user_query:
        raise RuntimeError("Query cannot be empty.")

    response = qa_chain.invoke({"query": user_query})
    print("RESULT:")
    print(response["result"])
    print("\nSOURCE DOCUMENTS:")
    print(format_source_documents(response.get("source_documents", [])))


if __name__ == "__main__":
    main()
