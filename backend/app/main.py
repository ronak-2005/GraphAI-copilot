from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.routes.auth import router as auth_router

app = FastAPI()

app.include_router(auth_router, prefix="/api/auth")

from supabase import create_client, Client
import os
from dotenv import load_dotenv
from app.supabase_client import supabase

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth")

@app.get("/")
def home():
    return {"message": "Backend is running 🚀"}


