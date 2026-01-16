import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from typing import List
from langchain_core.documents import Document


#document loading function
def load_all_documents(directory):
    """Loads both .txt and .pdf files from a directory."""
    docs = []
    loaders = {".txt": TextLoader, ".pdf": PyPDFLoader}
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        ext = os.path.splitext(file)[1].lower()
        
        if ext in loaders:
            try:
                loader = loaders[ext](file_path)
                docs.extend(loader.load())
                print(f"Successfully loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return docs

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

#text splitting function
def text_split(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
        )
    text_chunks = text_splitter.split_documents(docs)
    return text_chunks\
    

#embedding function
def download_embeddings():
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        
    )
    return embeddings

embedding = download_embeddings()