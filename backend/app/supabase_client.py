from supabase import create_client
from app.config import SUPABASE_URL, SUPABASE_KEY

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Supabase configuration missing! Check .env file.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
