import os

def setup_environment():
    os.system("pip install -Uqqq pip ")
    os.system("pip install -qqq langchain PyPDF2 gdown sentence-transformers faiss-cpu fastapi uvicorn transformers")
