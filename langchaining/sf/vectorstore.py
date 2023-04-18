import os
from typing import List

from langchain import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchaining.helpers import BASE_DIR, gpt_key

raw = str(BASE_DIR / "sf/raw")


def get_docs() -> List[Document]:
    files = [file for file in os.listdir(raw) if ".txt" in file]
    docs = []

    for file in files:
        with open(f"{raw}/{file}", "r") as f:
            # Extract the title for metadata
            title = f.read().split("\n")[0].strip()

            loader = TextLoader(f"{raw}/{file}")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )
            texts = text_splitter.split_documents(documents)
            for text in texts:
                text.metadata["episode_title"] = title
                # Replace multiple spaces with single to reduce length
                text.page_content = " ".join(text.page_content.split())
                docs.append(text)

    return docs


embeddings = OpenAIEmbeddings(
    document_model_name="text-embedding-ada-002",
    query_model_name="text-embedding-ada-002",
    openai_api_key=gpt_key,
)


def get_db() -> FAISS:
    if os.path.exists(f"{raw}/db"):
        print("DB already exists, loading")
        return FAISS.load_local(f"{raw}/db", embeddings=embeddings)
    else:
        print("Creating a new DB")
        db = FAISS.from_documents(get_docs(), embedding=embeddings)
        FAISS.save_local(db, f"{raw}/db")
        return db
