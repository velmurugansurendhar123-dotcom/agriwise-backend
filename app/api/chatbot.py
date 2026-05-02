from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import httpx

router = APIRouter()

SYSTEM_PROMPT = """You are AgriBot, an expert AI farming assistant for Indian farmers.
You help with crop selection, fertilizers, pest control, weather impact, market prices,
government schemes, irrigation, and organic farming in simple, practical language.
Always give quantities in kg/acre. Be friendly, encouraging and brief.
If asked in Hindi or Tamil, respond in that language."""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    language: str = "en"

FALLBACK_RESPONSES = {
    "rice": "🌾 Rice needs N=80-100, P=40-60, K=40-60 kg/ha. pH 5.5-7.0. Plant June-July. Yield: ~1500 kg/acre.",
    "wheat": "🌿 Wheat grows best 15-25°C. Sow November-December. Apply 120kg N/ha. Harvest April-May.",
    "fertilizer": "💊 Common fertilizers:\n• Urea (46%N) — 50kg/acre\n• DAP — 50kg/acre at sowing\n• MOP — 30kg/acre",
    "disease": "🔬 Common diseases:\n• Leaf Blight → Mancozeb 2g/L\n• Rust → Propiconazole 1ml/L\n• Mildew → Neem oil spray",
    "water": "💧 Irrigation tips:\n• Drip saves 40-60% water\n• Sandy soil: every 3-4 days\n• Clay soil: every 7-10 days",
    "scheme": "🏛️ Key schemes:\n• PM-KISAN: ₹6000/year\n• Fasal Bima: Crop insurance\n• KCC: 4% interest loan",
    "profit": "💰 To maximize profit:\n• Choose right crop for soil\n• Use quality seeds\n• Apply fertilizers on time\n• Plan irrigation well",
    "soil": "🌍 Soil pH guide:\n• 6.0-7.5 is ideal\n• Below 6: Add lime 200kg/acre\n• Above 7.5: Add gypsum 100kg/acre",
}

@router.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in (req.history or [])]
    messages.append({"role": "user", "content": req.message})

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "system": SYSTEM_PROMPT,
                    "messages": messages,
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                reply = data["content"][0]["text"]
                return {"reply": reply, "source": "ai"}
    except Exception:
        pass

    # Smart fallback
    msg_lower = req.message.lower()
    for keyword, response in FALLBACK_RESPONSES.items():
        if keyword in msg_lower:
            return {"reply": response, "source": "fallback"}

    return {
        "reply": "🤖 I'm AgriBot! Ask me about:\n• Crop selection\n• Fertilizers & dosage\n• Disease treatment\n• Government schemes\n• Profit estimation\n• Irrigation tips",
        "source": "fallback"
    }
