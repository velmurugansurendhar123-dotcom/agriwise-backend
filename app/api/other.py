from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from app.services.ml_service import ml_service
from app.services.database import get_db
from app.services.auth_service import get_current_user
import uuid

# ─── Disease Router ────────────────────────────────────────────────────────────
disease_router = APIRouter()

@disease_router.post("/detect_disease")
async def detect_disease(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    contents = await image.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
    return ml_service.detect_disease(contents)


# ─── Fertilizer Router ─────────────────────────────────────────────────────────
fertilizer_router = APIRouter()

class FertilizerInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    crop: str
    soil_type: str = "loamy"

@fertilizer_router.post("/fertilizer_recommend")
async def fertilizer_recommend(data: FertilizerInput):
    return ml_service.get_fertilizer_recommendation(
        data.nitrogen, data.phosphorus, data.potassium,
        data.ph, data.crop, data.soil_type)


# ─── Farm Router ────────────────────────────────────────────────────────────────
farm_router = APIRouter()

class FarmCreate(BaseModel):
    name: str
    location_name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    land_size_acres: float
    soil_type: str = "loamy"
    irrigation_available: bool = False

@farm_router.post("/create")
async def create_farm(farm: FarmCreate, user=Depends(get_current_user)):
    db = get_db()
    doc = {"_id": str(uuid.uuid4()), "owner_id": user["sub"],
           **farm.dict(), "created_at": datetime.utcnow().isoformat()}
    if db is not None:
        await db.farms.insert_one(doc)
    return {"id": doc["_id"], **farm.dict(), "message": "Farm created successfully"}

@farm_router.get("/list")
async def list_farms(user=Depends(get_current_user)):
    db = get_db()
    if db is not None:
        farms = await db.farms.find({"owner_id": user["sub"]}).to_list(100)
        for f in farms:
            f["id"] = f.pop("_id")
        return {"farms": farms}
    return {"farms": []}

@farm_router.delete("/{farm_id}")
async def delete_farm(farm_id: str, user=Depends(get_current_user)):
    db = get_db()
    if db is not None:
        await db.farms.delete_one({"_id": farm_id, "owner_id": user["sub"]})
    return {"message": "Farm deleted"}
