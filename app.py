# app.py
from fastapi import FastAPI, Depends, Cookie
from pydantic import BaseModel, Field
import local_embed_remote_llm as stylist  
from fastapi.middleware.cors import CORSMiddleware
from auth import authorize_user

app = FastAPI(title="Wardrobe Stylist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://nocatch.duckdns.org",
        "https://guemlaai.site",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    wardrobe: list | None = Field(default=None, description="Optional wardrobe payload sent from client cache")
    wardrobe_version: str | None = Field(default=None, description="Client-side wardrobe version for logging")

@app.post("/suggest-outfit")
async def suggest_outfit(req: QueryRequest, user=Depends(authorize_user)):
    # ✅ Lazy-load: only load models if not already loaded
    if getattr(stylist, "embed_model", None) is None:
        print("🧩 Lazy loading models (first request)...")
        stylist.embed_model = stylist.SentenceTransformer(
            stylist.EMBED_MODEL_PATH,
            device=stylist.DEVICE
        )
        print("✅ Models loaded successfully.")

    user_id = user["user_id"]

    # Prefer client-provided wardrobe to avoid re-downloading from API
    wardrobe_items = req.wardrobe if req.wardrobe else stylist.load_wardrobe_for_user(user_id)
    print("wardrobe_items:", wardrobe_items)
    # step 2: filter wardrobe

    # step 3: build faiss index from filtered wardrobe
    index, _ = stylist.build_faiss_index(wardrobe_items)
    
    # step 4: suggest outfit from filtered wardrobe
    result = stylist.suggest_outfit_from_wardrobe(req.query, wardrobe_items, index)

    return result


