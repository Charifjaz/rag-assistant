from fastapi import FastAPI
from app.rag_engine import ask_question
from app import config  # 👈 on importe la config ici

app = FastAPI()

@app.get("/")
def home():
    return {"message": "✅ API RAG en ligne. Utilisez /rag?question=..."}


@app.get("/rag")
def rag_query(
    q: str,
    model_name: str = config.DEFAULT_MODEL,
    temperature: float = config.DEFAULT_TEMPERATURE,
    k: int = config.DEFAULT_K
):
    return {
        "response": ask_question(
            question=q,
            model_name=model_name,
            temperature=temperature,
            k=k
        )
    }
