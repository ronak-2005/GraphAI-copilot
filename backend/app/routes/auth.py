from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client
import os

router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class AuthPayload(BaseModel):
    email: str
    password: str

@router.post("/signup")
def signup(payload: AuthPayload):
    res = supabase.auth.sign_up({
        "email": payload.email,
        "password": payload.password
    })
    
    if res.user is None:
        raise HTTPException(status_code=400, detail=res.error.message)
    
    return {"message": "Signup successful. Please check your email."}


@router.post("/login")
def login(payload: AuthPayload):
    res = supabase.auth.sign_in_with_password({
        "email": payload.email,
        "password": payload.password
    })

    if res.session is None:
        raise HTTPException(status_code=400, detail="Invalid email/password")

    return {
        "access_token": res.session.access_token,
        "refresh_token": res.session.refresh_token,
        "user": res.user
    }
