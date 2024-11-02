import os
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss

def download_documents(file_ids, output_dir="../data"):
    """
    Downloads multiple documents from Google Drive using gdown.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_id in file_ids:
        output_path = os.path.join(output_dir, f"{file_id}.pdf")
        os.system(f"gdown {file_id} -O {output_path}")
        print(f"Downloaded file to: {output_path}")


def parse_document_with_langchain(document_path):
    """
    Parses the document using LangChain.
    """
    loader = PyPDFLoader(document_path)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(document)
    return "\n\n".join([chunk.page_content for chunk in chunks])


def generate_embeddings(chunks):
    """
    Generate embeddings using a pre-trained transformer model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(chunks)

def index_embeddings(embeddings):
    """
    Store embeddings in a FAISS index for similarity search.
    """
    embedding_dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dimension)
    embeddings_array = np.array(embeddings).astype('float32')
    index.add(embeddings_array)
    return index