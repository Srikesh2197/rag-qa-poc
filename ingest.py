import os

from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def load_documents_from_folder(folder_path):
    docs = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Raw docs folder not found: {folder_path}")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())

        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            docs.extend(loader.load())

        else:
            print(f"Unsupported file type found, re-upload it .....")
            continue

    return docs

def ingest(reset_db):
    docs = load_documents_from_folder('data\\raw')
    if not docs:
        print(f"No supported documents uploaded/found. Add PDF/DOCX/TXT and re-upload .....")
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if reset_db and os.path.exists('data\\chroma'):
        import shutil
        shutil.rmtree('data\\chroma')

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory='data\\chroma',
    )

    vectordb.persist()

    return len(chunks)

if __name__ == "__main__":
    ingest(reset_db=True)