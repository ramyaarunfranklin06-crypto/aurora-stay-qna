from langchain_openai import ChatOpenAI
import os

def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)
    return [f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}" for r in results]

def generate(query, retrieved_chunks, model) -> str:
    prompt = f"""
    You are an AI assistant answering questions about medical insurance policies.
    User Query: {query}
    Retrieved Policy Information: {retrieved_chunks}
    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge.
    3. Include all relevant information.
    4. Do not omit important conditions.
    5. Focus only on relevant information.
    6. Do not include unrelated information.
    7. Answer the question directly.
    8. Keep it concise.
    9. Do not invent details.
    10. If insufficient, say "The available policy documents do not contain enough information to answer this question."
    Answer:
    """
    return model.invoke(prompt)

def rag(query: str, k: int = 2, model_name: str = "gpt-4o-mini", temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 512, vectorstore=None):
    model = ChatOpenAI(model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens, openai_api_key=os.getenv("OPENAI_API_KEY"), openai_api_base=os.getenv("OPENAI_API_BASE"))
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    answer = generate(query=query, retrieved_chunks=retrieved_chunks, model=model)
    return answer.content, retrieved_chunks
