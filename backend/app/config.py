import os
from dotenv import load_dotenv

# Load .env file when FastAPI starts
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ ERROR: Supabase URL or Key is missing.")
else:
    print("✅ Supabase config loaded.")
