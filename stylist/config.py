"""
Shared configuration and constants for the wardrobe stylist pipeline.
"""
import os
import torch
from dotenv import load_dotenv

# Load .env values, overriding any existing env vars so edits take effect without cleaning the shell
load_dotenv(override=True)

WHITELIST_TOKENS = [
    "outfit", "clothes", "clothing", "wardrobe", "attire", "wear", "wearing",
    "style", "fashion", "look", "fit",
    "shirt", "tshirt", "tee", "polo", "hoodie", "sweater", "cardigan", "jacket", "coat", "blazer", "vest",
    "pants", "trousers", "jeans", "shorts", "skirt", "dress", "suit", "tie", "bow tie",
    "shoes", "sneakers", "boots", "loafers", "sandals", "heels", "footwear",
    "bag", "belt",
    "formal", "casual", "business", "smart casual", "business casual",
    "wedding", "ceremony", "party", "interview", "presentation", "meeting", "office", "work",
    # Vietnamese tokens
    "mặc", "áo", "quần", "váy", "giày", "dép", "áo sơ mi", "quần tây", "áo khoác", "áo vest", "suit",
    "trang phục", "phong cách", "đi tiệc", "phỏng vấn", "đi chơi", "đi làm", "đi làm việc", "đi họp", "phòng họp",
    "bảo vệ đồ án", "bảo vệ luận văn", "thuyết trình", "thuyết trình khách hàng",
    "tiệc cưới", "dự cưới", "đám cưới", "đồ cưới", "dạ tiệc",
    "dã ngoại", "đi biển", "đi du lịch"
]

BLACKLIST_TOKENS = [
    "http://", "https://", "www.", ".com", ".net", ".org",
    "select ", "insert ", "update ", "delete ", "drop ", "union ",
    "sql ", "ssh ", "ftp ", "sudo", "rm -", "pip ", "npm ", "curl ",
    "password", "bank", "loan", "crypto", "bitcoin", "eth", "server",
    "politics", "election", "president", "prime minister"
]

EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_MODEL_PATH = "./models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

USE_CUDA = torch.cuda.is_available() and os.getenv("USE_CUDA", "1") == "1"
os.environ["HF_HUB_OFFLINE"] = "1"
DEVICE = "cuda" if USE_CUDA else "cpu"
TOP_K_EMBED = 25
TOP_K_RERANK = 7
QUERY_SCORE_AGGREGATION = "max"  # "max" or "mean" when merging multi-view query scores
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

WORKING_DIR = "wardrobe_data"
WARDROBE_API_BASE = os.getenv("WARDROBE_API_BASE", "https://localhost:7016")
API_URL = f"{WARDROBE_API_BASE}/item-descriptions/{{userId}}/raw"

print("OPENROUTER_KEY:", OPENROUTER_KEY)


def ensure_working_dir():
    """Ensure the working/cache directory exists."""
    if WORKING_DIR and not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR, exist_ok=True)


# Make sure the working directory is available at import-time, matching previous behavior.
ensure_working_dir()
