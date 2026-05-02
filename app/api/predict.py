from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.services.ml_service import ml_service

router = APIRouter()

class FarmInput(BaseModel):
    nitrogen: float = Field(default=70, ge=0, le=300)
    phosphorus: float = Field(default=40, ge=0, le=300)
    potassium: float = Field(default=45, ge=0, le=300)
    ph: float = Field(default=6.5, ge=0, le=14)
    temperature: float = Field(default=25, ge=-10, le=60)
    humidity: float = Field(default=70, ge=0, le=100)
    rainfall: float = Field(default=100, ge=0, le=5000)
    land_size_acres: float = Field(default=1.0, gt=0)
    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class SmartInput(BaseModel):
    latitude: float
    longitude: float
    land_size_acres: float = 1.0
    water_availability: str = "medium"  # low, medium, high
    season: str = "kharif"  # kharif, rabi, both

@router.post("/predict_crop")
async def predict_crop(data: FarmInput):
    if not ml_service.is_ready:
        raise HTTPException(status_code=503, detail="ML models loading, please wait")
    crops, importances = ml_service.predict_crops(
        data.nitrogen, data.phosphorus, data.potassium,
        data.ph, data.temperature, data.humidity, data.rainfall)
    summary = f"Based on your soil and climate, {crops[0]['crop'].title()} is your best option with {crops[0]['confidence']}% confidence."
    return {
        "top_crops": crops,
        "feature_importance": importances,
        "recommendation_summary": summary,
        "input_analysis": {
            "soil_health": "good" if 6 <= data.ph <= 7.5 else ("acidic" if data.ph < 6 else "alkaline"),
            "nutrient_balance": "balanced" if data.nitrogen > 40 and data.phosphorus > 30 else "needs_improvement",
            "water_availability": "adequate" if data.rainfall > 60 else "scarce",
            "climate_suitability": "favorable" if 15 <= data.temperature <= 35 else "challenging",
        }
    }

@router.post("/predict_smart")
async def predict_smart(data: SmartInput):
    """Smart mode: auto-fetch soil data from GPS location"""
    if not ml_service.is_ready:
        raise HTTPException(status_code=503, detail="ML models loading")

    # Get soil data from location
    soil = ml_service.get_soil_by_location(data.latitude, data.longitude)

    # Adjust for water availability
    water_map = {"low": 40, "medium": 100, "high": 200}
    rainfall_est = water_map.get(data.water_availability, 100)

    # Seasonal temperature adjustment
    season_temp = {"kharif": 28, "rabi": 18, "both": 24}
    temp_est = season_temp.get(data.season, 25)

    crops, importances = ml_service.predict_crops(
        soil["N"], soil["P"], soil["K"], soil["ph"],
        temp_est, 70, rainfall_est)

    return {
        "top_crops": crops,
        "feature_importance": importances,
        "auto_detected": {
            "region": soil["region"],
            "soil_nitrogen": soil["N"],
            "soil_phosphorus": soil["P"],
            "soil_potassium": soil["K"],
            "soil_ph": soil["ph"],
            "estimated_temperature": temp_est,
            "estimated_rainfall": rainfall_est,
        },
        "recommendation_summary": f"Based on {soil['region']} soil data, {crops[0]['crop'].title()} suits your farm best.",
    }

@router.post("/predict_profit")
async def predict_profit(data: FarmInput):
    if not ml_service.is_ready:
        raise HTTPException(status_code=503, detail="ML models loading")
    crops, _ = ml_service.predict_crops(
        data.nitrogen, data.phosphorus, data.potassium,
        data.ph, data.temperature, data.humidity, data.rainfall)
    predictions = ml_service.predict_profit(crops, data.land_size_acres)
    best = max(predictions, key=lambda x: x["net_profit"])
    return {
        "predictions": predictions,
        "best_crop": best["crop"],
        "currency": "INR",
        "note": f"Estimates for {data.land_size_acres} acres. Actual varies with market.",
    }

@router.post("/predict_risk")
async def predict_risk(data: FarmInput):
    if not ml_service.is_ready:
        raise HTTPException(status_code=503, detail="ML models loading")
    crops, _ = ml_service.predict_crops(
        data.nitrogen, data.phosphorus, data.potassium,
        data.ph, data.temperature, data.humidity, data.rainfall)
    top_crop = crops[0]["crop"] if crops else "rice"
    return ml_service.predict_risk(
        data.nitrogen, data.phosphorus, data.potassium,
        data.ph, data.temperature, data.humidity, data.rainfall, top_crop)

@router.post("/farming_plan")
async def farming_plan(data: FarmInput):
    if not ml_service.is_ready:
        raise HTTPException(status_code=503, detail="ML models loading")
    crops, _ = ml_service.predict_crops(
        data.nitrogen, data.phosphorus, data.potassium,
        data.ph, data.temperature, data.humidity, data.rainfall)
    crop = crops[0]["crop"] if crops else "rice"
    return ml_service.get_farming_plan(crop, data.land_size_acres)
