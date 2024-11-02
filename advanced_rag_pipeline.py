import os
from sentence_transformers import SentenceTransformer
from utils import download_documents
from utils import parse_document_with_langchain
from utils import generate_embeddings, index_embeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

def query_pipeline(pdf_path, user_query, model_path):
    """
    Full pipeline that takes user query and returns an answer.
    """
    # Step 1: Parse Document
    parsed_data = parse_document_with_langchain(pdf_path)
    chunks = parsed_data.split("\n\n")

    # Step 2: Generate Embeddings for Parsed Data
    embeddings = generate_embeddings(chunks)

    # Step 3: Index in FAISS
    index = index_embeddings(embeddings)

    # Step 4: Retrieve Relevant Context for Query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([user_query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k=5)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    context = " ".join(retrieved_chunks)

    # Step 5: Generate Answer Using the Local LLaMA Model
    prompt = create_prompt(context, user_query)
    answer = ask_llama_local(prompt, model_path)
    return answer

def create_prompt(context, question):
    """
    Create a structured prompt using the retrieved context and user question.
    """
    prompt_template = """
    You are an expert assistant. Use the following context to answer the user's question accurately.
    Context: {context}
    Question: {question}
    Provide a concise and accurate answer:
    """
    prompt = prompt_template.format(context=context, question=question)
    return prompt

def ask_llama_local(prompt, model_path):
    """
    Generate a response from a locally running LLaMA model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
