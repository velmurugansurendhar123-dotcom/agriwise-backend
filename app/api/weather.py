from fastapi import APIRouter, Query
import httpx, math
from app.config import settings

router = APIRouter()

@router.get("/get_weather")
async def get_weather(lat: float = Query(...), lon: float = Query(...)):
    if settings.OPENWEATHER_API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={settings.OPENWEATHER_API_KEY}&units=metric"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    d = resp.json()
                    return {
                        "temperature": d["main"]["temp"],
                        "feels_like": d["main"]["feels_like"],
                        "humidity": d["main"]["humidity"],
                        "description": d["weather"][0]["description"].title(),
                        "icon": d["weather"][0]["icon"],
                        "wind_speed": d["wind"]["speed"],
                        "city": d.get("name", "Unknown"),
                        "rainfall_mm": d.get("rain", {}).get("1h", 0),
                        "source": "live"
                    }
        except Exception:
            pass
    temp = round(28 - abs(lat - 20) * 0.4 + math.sin(lon/10)*2, 1)
    return {
        "temperature": temp,
        "feels_like": temp+2,
        "humidity": 72,
        "description": "Partly Cloudy",
        "icon": "02d",
        "wind_speed": 3.5,
        "city": "Your Location",
        "rainfall_mm": 2.1,
        "source": "estimated",
        "note": "Add OPENWEATHER_API_KEY for live data"
    }

@router.get("/get_forecast")
async def get_forecast(lat: float = Query(...), lon: float = Query(...)):
    if settings.OPENWEATHER_API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={settings.OPENWEATHER_API_KEY}&units=metric&cnt=5"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    return {"forecast": [
                        {"date": item["dt_txt"],
                         "temp": item["main"]["temp"],
                         "humidity": item["main"]["humidity"],
                         "description": item["weather"][0]["description"]}
                        for item in data["list"]], "source": "live"}
        except Exception:
            pass
    days = ["Mon","Tue","Wed","Thu","Fri"]
    return {"forecast": [
        {"date": days[i], "temp": 28+i,
         "humidity": 70+i, "description": "Partly Cloudy"}
        for i in range(5)], "source": "estimated"}