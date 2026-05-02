from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class Database:
    client: AsyncIOMotorClient = None
    db = None

db_instance = Database()

async def connect_db():
    try:
        db_instance.client = AsyncIOMotorClient(settings.MONGODB_URL)
        db_instance.db = db_instance.client[settings.DATABASE_NAME]
        await db_instance.client.admin.command("ping")
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.error(f"❌ MongoDB failed: {e}")

async def disconnect_db():
    if db_instance.client:
        db_instance.client.close()

def get_db():
    return db_instance.db