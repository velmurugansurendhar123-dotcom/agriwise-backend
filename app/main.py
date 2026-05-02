"""
AgriWise AI v2.0 - Smart Agriculture SaaS Platform
Complete Backend with ML, GeoAI, Schemes, Reports, OTP & Chatbot
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import logging

from app.api.auth import router as auth_router
from app.api.predict import router as predict_router
from app.api.weather import router as weather_router
from app.api.schemes import router as schemes_router
from app.api.report import router as report_router
from app.api.chatbot import router as chatbot_router
from app.api.other import disease_router, fertilizer_router, farm_router
from app.services.database import connect_db, disconnect_db
from app.services.ml_service import ml_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting AgriWise AI v2.0...")
    await connect_db()
    await ml_service.load_models()
    logger.info("✅ All services ready!")
    yield
    await disconnect_db()
    logger.info("👋 AgriWise AI v2.0 shut down.")


app = FastAPI(
    title="AgriWise AI API v2",
    description="""
    🌾 **AgriWise AI** - Smart Agriculture SaaS Platform
    
    Complete backend with:
    - 🤖 AI Crop Prediction (RandomForest ML)
    - 💰 Profit Estimation
    - ⚠️ Risk Analysis  
    - 📍 GeoAI Smart Mode (GPS-based)
    - 🏛️ Government Schemes Finder
    - 📄 PDF Report Generation
    - 🌤️ Live Weather
    - 🔬 Disease Detection
    - 🤖 AI Chatbot
    - 💊 Fertilizer Recommendations
    - 📱 OTP Login
    """,
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount all routers
app.include_router(auth_router,        prefix="/api/v1/auth",       tags=["🔐 Authentication"])
app.include_router(predict_router,     prefix="/api/v1",            tags=["🌾 AI Predictions"])
app.include_router(weather_router,     prefix="/api/v1",            tags=["🌤️ Weather"])
app.include_router(schemes_router,     prefix="/api/v1",            tags=["🏛️ Govt Schemes"])
app.include_router(report_router,      prefix="/api/v1",            tags=["📄 Reports"])
app.include_router(chatbot_router,     prefix="/api/v1",            tags=["🤖 AI Chatbot"])
app.include_router(disease_router,     prefix="/api/v1",            tags=["🔬 Disease Detection"])
app.include_router(fertilizer_router,  prefix="/api/v1",            tags=["💊 Fertilizer"])
app.include_router(farm_router,        prefix="/api/v1/farm",       tags=["🏡 Farm Management"])


@app.get("/", tags=["System"])
async def root():
    return {
        "app": "AgriWise AI v2.0",
        "status": "running 🌾",
        "docs": "/docs",
        "version": "2.0.0",
        "features": [
            "AI Crop Prediction", "Profit Estimation", "Risk Analysis",
            "GeoAI Smart Mode", "Government Schemes", "PDF Reports",
            "Live Weather", "Disease Detection", "AI Chatbot",
            "Fertilizer Recommendations", "OTP Login"
        ]
    }

@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy", "ml_ready": ml_service.is_ready, "version": "2.0.0"}
