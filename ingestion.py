from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_all_documents, text_split, download_embeddings
from dotenv import load_dotenv
import os

INDEX_NAME = "fitness-research"
RESEARCH_PATH = "data/knowledge_base/"

load_dotenv()

PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_all_documents(RESEARCH_PATH)
text_chunks = text_split(extracted_data)
embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384, # Dimension for 'all-MiniLM-L6-v2'
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)