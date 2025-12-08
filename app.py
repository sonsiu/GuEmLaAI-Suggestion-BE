# app.py
from fastapi import FastAPI, Depends, Cookie, HTTPException
from pydantic import BaseModel, Field
import local_embed_remote_llm as stylist  
from fastapi.middleware.cors import CORSMiddleware
from auth import authorize_user
import os
import requests

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

# Upstream API used to proxy profile requests (falls back to wardrobe API base)
PROFILE_API_BASE = os.getenv("PROFILE_API_BASE", os.getenv("WARDROBE_API_BASE", "https://localhost:7016/api"))
PROFILE_ENDPOINT = f"{PROFILE_API_BASE.rstrip('/')}/UserProfile/profile"

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

    # step 3: build faiss index from filtered wardrobe (cached per user/version)
    index, _ = stylist.build_faiss_index(
        wardrobe_items,
        user_id=user_id,
        wardrobe_version=req.wardrobe_version,
    )
    
    # step 4: suggest outfit from filtered wardrobe
    result = stylist.suggest_outfit_from_wardrobe(req.query, wardrobe_items, index)

    return result

@app.get("/UserProfile/profile")
async def proxy_user_profile(user=Depends(authorize_user)):
    """
    Proxy the user profile endpoint to the main backend so the frontend can call this service directly.
    """
    token = user.get("token")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        resp = requests.get(PROFILE_ENDPOINT, headers=headers, timeout=10, verify=False)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to reach profile service: {exc}")

    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    try:
        return resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Profile service returned non-JSON response")

