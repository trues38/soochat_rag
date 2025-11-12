from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import torch
import os

# ================================================
# 🧩 기본 설정
# ================================================
app = FastAPI(title="SooChat HuggingFace RAG API")

# CORS (Dify, 웹앱, Render 클라이언트와 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================
# 🤖 HuggingFace 모델 로드
# ================================================
MODEL_NAME = os.getenv("HF_MODEL", "jhgan/ko-sroberta-multitask")  # 한국어 RAG용
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

# 임시 벡터 저장소 (메모리 기반)
vector_store = []

print(f"✅ HuggingFace model loaded: {MODEL_NAME} ({device})")


# ================================================
# 🧭 헬스체크
# ================================================
@app.get("/")
async def root():
    return {"status": "ok", "message": "SooChat HuggingFace RAG Server running 🚀"}


# ================================================
# 📝 벡터 업서트 (Insert/Update)
# ================================================
@app.post("/upsert")
async def upsert_vectors(request: Request):
    body = await request.json()
    texts = body.get("texts", [])
    meta = body.get("meta", {})

    if not texts:
        return {"error": "No texts provided"}

    embeddings = model.encode(texts, convert_to_tensor=True)
    for i, text in enumerate(texts):
        vector_store.append({
            "text": text,
            "embedding": embeddings[i],
            "meta": meta
        })

    return {"status": "success", "count": len(texts)}


# ================================================
# 🔍 벡터 검색
# ================================================
@app.post("/query")
async def query_vectors(request: Request):
    body = await request.json()
    query = body.get("query")
    limit = int(body.get("limit", 3))

    if not query:
        return {"error": "query missing"}

    if not vector_store:
        return {"error": "vector store empty"}

    query_emb = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = torch.stack([v["embedding"] for v in vector_store])

    cosine_scores = util.cos_sim(query_emb, corpus_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=min(limit, len(vector_store)))

    results = []
    for idx, score in zip(top_results.indices, top_results.values):
        item = vector_store[idx]
        results.append({
            "text": item["text"],
            "score": float(score),
            "meta": item.get("meta", {})
        })

    return {"query": query, "results": results}


# ================================================
# ⚙️ 서버 실행 정보 출력
# ================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))