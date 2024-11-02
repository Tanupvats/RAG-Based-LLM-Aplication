from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from advanced_rag_pipeline import query_pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

model = None
tokenizer = None

class QueryRequest(BaseModel):
    pdf_path: str
    user_query: str

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model_path = "./models/fine_tuned_llama_model"
    try:
        # Load the tokenizer and model at startup
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Failed to load the model")    

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        answer = query_pipeline(request.pdf_path, request.user_query, "../models/fine_tuned_llama_model")
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
