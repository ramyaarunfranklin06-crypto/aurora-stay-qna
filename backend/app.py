from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag_utils import retrieve, rag

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = "faiss_index"

rag_api = Flask("Aurora Stay API")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

@rag_api.get("/")
def home():
    return "Welcome to the Aurora Stay RAG API!"

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    data = request.get_json()
    return jsonify({"query": data["query"], "k": data.get("k", 2), "relevant_chunks": retrieve(query=data["query"], k=data.get("k", 2), vectorstore=vectorstore)})

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    data = request.get_json()
    answer, chunks = rag(query=data["query"], k=data.get("k", 2), model_name=data.get("model_name", "gpt-4o-mini"), temperature=data.get("temperature", 0.0), top_p=data.get("top_p", 1.0), max_tokens=data.get("max_tokens", 512), vectorstore=vectorstore)
    return jsonify({"query": data["query"], "k": data.get("k", 2), "relevant_chunks": chunks, "answer": answer})

@rag_api.post("/v1/answer")
def answer():
    data = request.get_json()
    generated_answer, _ = rag(query=data["query"], k=data.get("k", 2), model_name=data.get("model_name", "gpt-4o-mini"), temperature=data.get("temperature", 0.0), top_p=data.get("top_p", 1.0), max_tokens=data.get("max_tokens", 512), vectorstore=vectorstore)
    return jsonify({"answer": generated_answer})

if __name__ == "__main__":
    rag_api.run(debug=True)
