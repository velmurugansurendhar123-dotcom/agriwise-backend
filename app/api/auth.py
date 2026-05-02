from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from app.services.database import get_db
from app.services.auth_service import hash_password, verify_password, create_access_token, generate_otp
import uuid

router = APIRouter()

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    phone: str = ""
    language: str = "en"

class LoginRequest(BaseModel):
    email: str
    password: str

class OTPRequest(BaseModel):
    phone: str

class OTPVerify(BaseModel):
    phone: str
    otp: str

# In-memory OTP store (use Redis in production)
otp_store = {}

@router.post("/register")
async def register(req: RegisterRequest):
    db = get_db()
    if db is not None:
        existing = await db.users.find_one({"email": req.email})
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        user = {
            "_id": str(uuid.uuid4()), "name": req.name, "email": req.email,
            "password": hash_password(req.password), "phone": req.phone,
            "language": req.language, "created_at": datetime.utcnow().isoformat()
        }
        await db.users.insert_one(user)
        user_id = user["_id"]
    else:
        user_id = str(uuid.uuid4())
    token = create_access_token({"sub": user_id, "email": req.email, "name": req.name})
    return {"access_token": token, "token_type": "bearer",
            "user": {"id": user_id, "name": req.name, "email": req.email, "language": req.language}}

@router.post("/login")
async def login(req: LoginRequest):
    db = get_db()
    if db is not None:
        user = await db.users.find_one({"email": req.email})
        if not user or not verify_password(req.password, user["password"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        token = create_access_token({"sub": user["_id"], "email": user["email"], "name": user["name"]})
        return {"access_token": token, "token_type": "bearer",
                "user": {"id": user["_id"], "name": user["name"], "email": user["email"]}}
    token = create_access_token({"sub": "demo", "email": req.email, "name": "Demo Farmer"})
    return {"access_token": token, "token_type": "bearer",
            "user": {"id": "demo", "name": "Demo Farmer", "email": req.email}}

@router.post("/send_otp")
async def send_otp(req: OTPRequest):
    otp = generate_otp()
    otp_store[req.phone] = {"otp": otp, "created_at": datetime.utcnow().isoformat()}
    # In production: send via Twilio SMS
    # For demo: return OTP directly
    return {"message": f"OTP sent to {req.phone}", "demo_otp": otp, "note": "In production OTP is sent via SMS"}

@router.post("/verify_otp")
async def verify_otp(req: OTPVerify):
    stored = otp_store.get(req.phone)
    if not stored or stored["otp"] != req.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    del otp_store[req.phone]
    user_id = str(uuid.uuid4())
    token = create_access_token({"sub": user_id, "phone": req.phone, "name": "Farmer"})
    return {"access_token": token, "token_type": "bearer",
            "user": {"id": user_id, "name": "Farmer", "phone": req.phone}}

@router.post("/guest")
async def guest_login():
    token = create_access_token({"sub": "guest", "name": "Guest Farmer", "email": "guest@agriwise.ai"})
    return {"access_token": token, "token_type": "bearer",
            "user": {"id": "guest", "name": "Guest Farmer", "email": "guest@agriwise.ai"}}
