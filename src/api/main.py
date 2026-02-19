from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.chatbot import ask_ollama
from src.api.schemas import SymptomRequest
from src.api.predictor import predict_top3
from src.api.utils import detect_language
from src.api.vaccine import vaccine_response

app = FastAPI(title="Disease Chatbot API")

# Static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Templates
templates = Jinja2Templates(directory="src/templates")

CONFIDENCE_THRESHOLD = 0.60

SYMPTOM_KEYWORDS = [
    "fever", "headache", "vomiting", "nausea", "pain",
    "chills", "rash", "cough", "fatigue", "diarrhea",
    "body pain", "joint pain", "sore throat"
]

# Health Check
@app.get("/health")
def health():
    return {"status": "ok"}


# Home Page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Main Chat Endpoint
@app.post("/chat")
def chat(request: SymptomRequest):

    text = request.text.lower().strip()
    language = detect_language(text)

    # 1️⃣ Greeting
    if text in ["hi", "hello", "hey"]:
        return {
            "mode": "conversation",
            "message": "Hello! I'm your AI health assistant. You can describe symptoms or provide a baby's birth date for vaccine tracking.",
            "language": language
        }

    # 2️⃣ Vaccine Module (MUST run before ML & LLM)
    vaccine_reply = vaccine_response(request.text)
    if vaccine_reply:
        return {
            "mode": "vaccine_schedule",
            "message": vaccine_reply,
            "language": language
        }

    # 3️⃣ Symptom Detection → ML Model
    if any(keyword in text for keyword in SYMPTOM_KEYWORDS):

        top3 = predict_top3(text)
        top_disease, top_confidence = top3[0]

        if float(top_confidence) >= CONFIDENCE_THRESHOLD:
            return {
                "mode": "ml_prediction",
                "disease": top_disease,
                "confidence": float(top_confidence),
                "message": (
                    f"Based on your symptoms, our AI model predicts "
                    f"{top_disease} with {float(top_confidence)*100:.2f}% confidence.\n\n"
                    "This is a preliminary assessment. Please consult a qualified doctor for confirmation."
                ),
                "language": language
            }

        else:
            reply = ask_ollama(request.text)
            return {
                "mode": "llm_followup",
                "message": reply,
                "language": language
            }

    # 4️⃣ Fallback → Conversational AI
    reply = ask_ollama(request.text)
    return {
        "mode": "llm_fallback",
        "message": reply,
        "language": language
    }
