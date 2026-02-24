from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from src.api.vaccine import vaccine_tracker
from src.api.predictor import predict_top3
from src.api.chatbot import ask_ollama

app = FastAPI(title="Disease Chatbot API")

# --------------------------------------------------
# CORS (Safe for local dev)
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Static & Templates
# --------------------------------------------------

app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")

# --------------------------------------------------
# Config
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.60

SYMPTOM_KEYWORDS = [
    "fever", "headache", "vomiting", "nausea", "pain",
    "chills", "rash", "cough", "fatigue", "diarrhea",
    "body pain", "joint pain", "sore throat"
]

# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --------------------------------------------------
# Main Chat Endpoint
# --------------------------------------------------

@app.post("/chat")
async def chat(request: Request):

    try:
        data = await request.json()
    except:
        return {"mode": "error", "message": "Invalid JSON received."}

    # Extract any string field safely
    user_input = next(
        (v for v in data.values() if isinstance(v, str) and v.strip()),
        ""
    ).strip()

    if not user_input:
        return {"mode": "error", "message": "Empty message received."}

    # --------------------------------------------------
    # 1️⃣ Vaccine Tracking (Highest Priority)
    # --------------------------------------------------

    vaccine_data = vaccine_tracker(user_input)

    if vaccine_data:
        return {
            "mode": "vaccine_tracking",
            "data": vaccine_data
        }

    # --------------------------------------------------
    # 2️⃣ Symptom-Based ML Prediction
    # --------------------------------------------------

    text = user_input.lower()

    if any(keyword in text for keyword in SYMPTOM_KEYWORDS):

        try:
            top3 = predict_top3(user_input)

            if top3:
                top_disease, top_confidence = top3[0]

                if float(top_confidence) >= CONFIDENCE_THRESHOLD:
                    return {
                        "mode": "ml_prediction",
                        "disease": top_disease.strip(),
                        "confidence": float(top_confidence),
                        "message": (
                            f"AI predicts {top_disease.strip()} "
                            f"with {float(top_confidence) * 100:.2f}% confidence.\n\n"
                            "⚠ This is a preliminary assessment. "
                            "Please consult a qualified doctor."
                        )
                    }

        except Exception as e:
            print("ML prediction error:", e)

    # --------------------------------------------------
    # 3️⃣ Fallback to Conversational AI
    # --------------------------------------------------

    try:
        reply = ask_ollama(user_input)

        if not reply:
            reply = "I'm here to help. Could you please provide more details?"

        return {
            "mode": "llm_fallback",
            "message": reply
        }

    except Exception as e:
        print("LLM error:", e)
        return {
            "mode": "error",
            "message": "System temporarily unavailable."
        }