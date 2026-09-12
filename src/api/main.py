import re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from src.api.vaccine import vaccine_tracker
from src.api.predictor import predict_top3
from src.api.chatbot import ask_ollama, ask_groq_medical

app = FastAPI(title="AI Health Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")

CONFIDENCE_THRESHOLD = 0.90
LLM_THRESHOLD = 0.40
MINIMUM_DISPLAY_CONFIDENCE = 0.20

# --------------------------------------------------
# Keyword Lists
# --------------------------------------------------

SYMPTOM_KEYWORDS = [
    # --- Core symptoms (exact) ---
    "fever", "headache", "head ache", "vomiting", "nausea", "pain",
    "chills", "rash", "cough", "fatigue", "diarrhea", "diarrhoea",
    "body pain", "bodypain", "joint pain", "jointpain", "sore throat",
    "sorethroat", "dizziness", "dizzy", "abdominal pain", "weakness",
    "weak", "sneeze", "sneezing", "runny nose", "congestion",
    "shortness of breath", "breathless", "swelling", "itching",
    "burning", "discharge", "bleeding", "chest pain", "back pain",
    "stomach pain", "muscle pain", "sweating", "shivering",
    "loss of appetite", "weight loss", "weight gain", "palpitations",
    "numbness", "tingling", "blurred vision", "difficulty swallowing",
    "dry mouth", "excessive thirst", "frequent urination", "insomnia",
    "anxiety", "depression", "confusion", "memory loss",
    # --- Natural language variants ---
    # Head / nausea  (FIX: "my head hurts" and "nauseous" were not matching)
    "head hurts", "head is hurting", "head is pounding", "head is spinning",
    "nauseou", "feel sick", "feeling sick", "feel like vomiting",
    "want to vomit", "throwing up",
    # Fatigue / weakness
    "tired", "fatigued", "exhausted", "lethargic", "no energy",
    "feel weak", "feeling weak", "feel tired", "feeling tired",
    # Stomach / digestion
    "stomach hurts", "stomach is hurting", "stomach ache", "stomachache",
    "tummy pain", "tummy ache", "belly pain", "belly ache",
    "loose stools", "loose motion", "loose motions", "watery stool",
    "upset stomach",
    # Breathing
    "hard to breathe", "trouble breathing", "breathing difficulty",
    "short of breath", "out of breath",
    # Chest
    "chest hurts", "chest is hurting", "chest tightness", "tight chest",
    "chest pressure", "heart racing", "heart pounding",
    # Skin / eyes
    "yellow skin", "yellow eyes", "dark urine", "jaundice",
    "skin is yellow", "eyes are yellow", "itchy skin", "skin rash",
    "red spots", "red patches", "skin peeling",
    # Hair / urinary
    "hair loss", "hair falling", "losing hair", "hair is falling",
    "urinating frequently", "urinate often", "frequent urge to urinate",
    "burning when urinating", "burning urination",
    # Thirst / hunger
    "thirsty", "always thirsty", "constant thirst",
    "constant hunger", "always hungry",
    # Temperature
    "high temperature", "high fever", "low grade fever", "feeling hot",
    "feeling cold", "cold and shivering",
    # General pain variants
    "body ache", "body aches", "muscle ache", "muscle aches",
    "joint ache", "back ache", "backache",
    # Vision / neurological
    "blurry vision", "double vision", "vision problems",
    "memory problems", "forgetting things", "brain fog",
    # Throat / mouth
    "throat hurts", "throat is sore", "trouble swallowing",
    # Sleep / mental
    "trouble sleeping", "not sleeping", "sleep problems",
    "feeling anxious", "feeling depressed", "mood swings",
    # Misc
    "pale skin", "pale face", "skin is pale",
    "bloated", "bloating",
    "stiff neck", "neck stiffness", "neck pain",
    "ear pain", "earache", "ear ache", "ringing in ears",
    "nose bleeding", "nosebleed",
]

VAGUE_PATTERNS = [
    "brain is melting", "feel weird", "feel strange", "body feels",
    "something is wrong", "feel off", "not feeling myself",
    "feel funny", "feel odd", "feel terrible", "feel awful",
    "feel bad", "not well", "feeling low", "feeling down",
    "i don't know", "not sure", "something is off",
    "not right", "off today"
]

NON_MEDICAL_PATTERNS = [
    "hello", "hi", "hey", "how are you", "what can you do",
    "who are you", "what are you", "good morning",
    "good evening", "good afternoon", "what is your name",
    "tell me about yourself"
]

# FIX: Expanded — apostrophe-stripped normalize_text now catches "can't" variants
EMERGENCY_SYMPTOMS = [
    # Breathing
    "difficulty breathing",
    "cannot breathe",
    "cant breathe",
    "cant breathe",
    "unable to breathe",
    "not able to breathe",
    "struggling to breathe",
    "stopped breathing",
    "chest is crushing",
    "crushing chest",
    "chest tightness",
    "chest feels tight",
    "chest pressure",
    # Cardiac
    "chest pain shortness",
    "chest pain sweating",
    "severe chest pain",
    "heart attack",
    "my heart is stopping",
    # Neurological
    "stroke symptoms",
    "face drooping",
    "arm weakness sudden",
    "speech difficulty sudden",
    "sudden confusion",
    "sudden severe headache",
    "worst headache of my life",
    # Consciousness
    "loss of consciousness",
    "unconscious",
    "collapsed",
    "passing out",
    "fainted",
    "unresponsive",
    # Seizure / Anaphylaxis
    "seizure",
    "anaphylaxis",
    "severe allergic",
    "throat is closing",
    "throat closing",
    "cant swallow",
    # Safety
    "want to die",
    "going to kill myself",
    "going to hurt myself",
]

DANGER_KEYWORDS = [
    "dangerous", "serious", "severe", "fatal", "deadly",
    "should i worry", "is it bad", "how bad", "emergency",
    "am i going to die", "will i die", "going to die",
    "is it fatal", "is it deadly", "life threatening",
    "should i go to hospital", "need to go to hospital",
    "is it serious", "how serious", "critical",
    "am i in danger", "should i be worried", "is it contagious",
    "will it spread", "can it kill", "is it curable",
    "how dangerous", "is it safe", "should i panic"
]

TREATMENT_KEYWORDS = [
    "treatment", "medicine", "medication", "cure", "remedy",
    "what should i do", "how to treat", "what to take", "doctor",
    "hospital", "clinic", "pharmacy", "prescription",
    "pills", "injection", "how to recover", "recover",
    "get better", "heal", "manage", "relief", "help me"
]

CAUSE_KEYWORDS = [
    "cause", "why", "how did", "reason", "how do i get",
    "what causes", "origin", "how does it spread", "spread",
    "how did i get", "where did", "source", "risk factor",
    "who gets", "who is at risk", "prone to"
]

SYMPTOM_MORE_KEYWORDS = [
    "what are the symptoms", "other symptoms", "more symptoms",
    "what else", "signs", "what to expect", "what will happen",
    "how will i feel", "what does it feel like", "symptoms of"
]

conversation_store: dict[str, dict] = {}

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def normalize_text(text: str) -> str:
    """
    FIX: Strip apostrophes/smart-quotes before keyword matching so that
    "can't breathe" correctly matches "cant breathe" in EMERGENCY_SYMPTOMS.
    """
    text = re.sub(r"['\u2018\u2019\u201c\u201d]", "", text)
    return " ".join(text.lower().split())


def detect_symptoms(text: str) -> list[str]:
    return [kw for kw in SYMPTOM_KEYWORDS if kw in text]


def is_emergency(text: str) -> bool:
    return any(p in text for p in EMERGENCY_SYMPTOMS)


def is_chitchat(text: str) -> bool:
    has_symptoms = bool(detect_symptoms(text))
    if has_symptoms:
        return False
    return any(p in text for p in NON_MEDICAL_PATTERNS)


def is_too_vague(text: str) -> bool:
    
    # Block explicit vague emotional phrases
    if any(p in text for p in VAGUE_PATTERNS):
        return True

    # Symptom detected — input is specific enough, never block
    if detect_symptoms(text):
        return False

    # No symptom keywords AND near-empty content → block
    content_words = [
        w for w in text.split()
        if w not in {"i", "am", "are", "have", "feel", "a", "the",
                     "my", "me", "it", "is", "was", "been", "im"}
    ]
    return len(content_words) <= 1


def build_context_text(history: list[dict], current_input: str, session: dict) -> str:

    if not history:
        return current_input

    current_state = session.get("state", "INITIAL")
    if current_state in ("INITIAL", "PREDICTED", "POST_PRED"):
        return current_input

    last_assistant = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"),
        None
    )

    is_followup = last_assistant and any(
        phrase in last_assistant.lower()
        for phrase in [
            "could you provide", "please describe",
            "additional details", "more detail",
            "how long have you", "how severe",
            "please answer", "to better assess",
            "please answer the following"
        ]
    )

    if is_followup:
        prev_user_turns = [m["content"] for m in history if m["role"] == "user"]
        recent = prev_user_turns[-2:]
        return " ".join(recent + [current_input])

    return current_input


def get_session(session_id: str) -> dict:
    if session_id not in conversation_store:
        conversation_store[session_id] = {
            "history": [],
            "state": "INITIAL",
            "last_disease": None,
            "last_confidence": None,
            "clarification_count": 0,
        }
    return conversation_store[session_id]


def reset_session(session_id: str):
    conversation_store[session_id] = {
        "history": [],
        "state": "INITIAL",
        "last_disease": None,
        "last_confidence": None,
        "clarification_count": 0,
    }


def handle_post_prediction(text: str, disease: str, confidence: float) -> str | None:

    if any(kw in text for kw in DANGER_KEYWORDS):
        high_risk = [
            "dengue", "malaria", "typhoid", "tuberculosis",
            "pneumonia", "heart attack", "stroke", "appendicitis",
            "pulmonary embolism", "meningitis", "hepatitis",
            "aids", "cancer", "sepsis"
        ]
        if any(d in disease.lower() for d in high_risk):
            return (
                f"{disease} can be serious if left untreated. "
                "Symptoms can worsen rapidly — please consult a doctor "
                "promptly or visit the nearest clinic. "
                "Do not self-medicate.\n\n"
                "If you are experiencing severe symptoms like difficulty "
                "breathing, chest pain, or loss of consciousness, "
                "seek emergency care immediately."
            )
        else:
            return (
                f"{disease} is generally manageable with proper treatment. "
                "However, you should still consult a qualified doctor "
                "for accurate diagnosis and appropriate care.\n\n"
                "Early treatment leads to faster recovery. "
                "Please do not ignore your symptoms."
            )

    if any(kw in text for kw in TREATMENT_KEYWORDS):
        return (
            f"For {disease}, treatment typically involves consulting "
            "a doctor who will prescribe appropriate medication "
            "based on your specific condition.\n\n"
            "General first steps:\n"
            "- Rest and stay hydrated\n"
            "- Do not self-medicate\n"
            "- Visit a clinic or hospital for proper diagnosis\n"
            "- Follow the doctor's prescription strictly\n\n"
            "Early medical attention leads to faster and safer recovery."
        )

    if any(kw in text for kw in CAUSE_KEYWORDS):
        return (
            f"{disease} can be caused by various factors including "
            "infections, environmental triggers, lifestyle factors, "
            "or underlying conditions.\n\n"
            "A qualified doctor can run the necessary tests to identify "
            "the specific cause in your case and recommend "
            "appropriate treatment."
        )

    if any(kw in text for kw in SYMPTOM_MORE_KEYWORDS):
        return (
            f"Common symptoms associated with {disease} may vary by "
            "individual and can change as the condition progresses.\n\n"
            "Please consult a doctor or check reliable medical sources "
            "like WHO (who.int) or NHS (nhs.uk) for a complete "
            "symptom list.\n\n"
            "If your symptoms are worsening or new symptoms appear, "
            "seek medical attention immediately."
        )

    if any(word in text for word in [
        "thank", "thanks", "ok", "okay", "noted", "got it",
        "alright", "understood", "i see", "i understand"
    ]):
        return (
            "You're welcome. Remember this is an AI assessment only — "
            "please consult a qualified doctor for accurate diagnosis "
            "and treatment.\n\nTake care and stay healthy!"
        )

    return None


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
    except Exception:
        return {"mode": "error", "message": "Invalid JSON received."}

    user_input = next(
        (v for v in data.values() if isinstance(v, str) and v.strip()),
        ""
    ).strip()

    if not user_input:
        return {"mode": "error", "message": "Empty message received."}

    session_id = data.get("session_id", "default")
    session = get_session(session_id)

    history = session["history"]
    history.append({"role": "user", "content": user_input})

    text = normalize_text(user_input)

    # --------------------------------------------------
    # 1. Emergency Triage
    # --------------------------------------------------

    if is_emergency(text):
        msg = (
            "⚠ EMERGENCY ALERT ⚠\n\n"
            "Your symptoms may indicate a medical emergency.\n\n"
            "Please SEEK EMERGENCY CARE IMMEDIATELY:\n"
            "- Call emergency services (108 / 112)\n"
            "- Go to the nearest emergency room\n"
            "- Do not wait or self-medicate\n\n"
            "Do not rely on AI assessment for emergency symptoms."
        )
        reply = {"mode": "emergency", "message": msg}
        history.append({"role": "assistant", "content": msg})
        return reply

    # --------------------------------------------------
    # 2. Vaccine Tracking
    # --------------------------------------------------

    vaccine_data = vaccine_tracker(user_input)
    if vaccine_data:
        reply = {"mode": "vaccine_tracking", "data": vaccine_data}
        history.append({"role": "assistant", "content": str(reply)})
        return reply

    # --------------------------------------------------
    # 3. Chitchat Guardrail
    # --------------------------------------------------

    if is_chitchat(text):
        try:
            reply_text = ask_ollama(user_input)
            if not reply_text:
                reply_text = (
                    "Hello! I'm your AI Health Assistant. "
                    "Please describe your symptoms and I'll help analyze them."
                )
        except Exception:
            reply_text = (
                "Hello! I'm your AI Health Assistant. "
                "Please describe your symptoms and I'll help analyze them."
            )
        reply = {"mode": "llm_fallback", "message": reply_text}
        history.append({"role": "assistant", "content": reply_text})
        return reply

    # --------------------------------------------------
    # 4. Conversational State Machine
    # --------------------------------------------------

    current_state = session["state"]

    if current_state == "PREDICTED" and session["last_disease"]:

        disease = session["last_disease"]
        confidence = session["last_confidence"]

        follow_up_response = handle_post_prediction(text, disease, confidence)

        if follow_up_response:
            if any(word in text for word in [
                "thank", "thanks", "ok", "okay", "noted",
                "got it", "alright", "understood"
            ]):
                session["state"] = "POST_PRED"
            history.append({"role": "assistant", "content": follow_up_response})
            return {
                "mode": "post_prediction",
                "disease": disease,
                "message": follow_up_response
            }

        new_symptoms = detect_symptoms(text)
        if new_symptoms:
            # FIX: reset to INITIAL so build_context_text doesn't carry
            # old symptom history into the new prediction embedding
            session["state"] = "INITIAL"
            session["last_disease"] = None
            session["last_confidence"] = None
            session["clarification_count"] = 0
        else:
            msg = (
                f"I've already assessed your symptoms as possibly {disease}. "
                "You can ask me:\n"
                "- Is this dangerous?\n"
                "- What is the treatment?\n"
                "- What causes this?\n\n"
                "Or describe new symptoms for a fresh assessment."
            )
            history.append({"role": "assistant", "content": msg})
            return {"mode": "clarification", "message": msg}

    if current_state == "POST_PRED":
        new_symptoms = detect_symptoms(text)
        if new_symptoms:
            session["state"] = "INITIAL"
            session["last_disease"] = None
            session["last_confidence"] = None
            session["clarification_count"] = 0
        elif any(word in text for word in [
            "thank", "thanks", "ok", "okay", "bye",
            "goodbye", "noted", "alright"
        ]):
            msg = (
                "Take care! If your symptoms persist or worsen, "
                "please visit a doctor. Stay healthy!"
            )
            history.append({"role": "assistant", "content": msg})
            reset_session(session_id)
            return {"mode": "clarification", "message": msg}

    # --------------------------------------------------
    # 5. Conversational Guardrail (before MPNet)
    # --------------------------------------------------

    context_text = build_context_text(history[:-1], user_input, session)
    context_normalized = normalize_text(context_text)

    current_symptoms = detect_symptoms(text)
    context_symptoms = detect_symptoms(context_normalized)

    has_symptoms = len(current_symptoms) >= 1 or len(context_symptoms) >= 2

    in_clarification_flow = (
        session.get("clarification_count", 0) > 0
        and session.get("pending_disease") is not None
    )

    if in_clarification_flow:
        has_symptoms = True
    elif has_symptoms and is_too_vague(text):
        has_symptoms = False

    if not has_symptoms and not in_clarification_flow and not any(
        word in text for word in ["feel", "sick", "unwell", "ill", "hurt", "ache"]
    ):
        msg = (
            "I can help analyze health symptoms. "
            "Please describe what you're experiencing, for example:\n\n"
            '- "I have fever, headache and body ache"\n'
            '- "Stomach pain with vomiting since 2 days"\n'
            '- "Cough, sore throat and mild fever"\n\n'
            "The more specific you are, the more accurate "
            "my assessment will be."
        )
        reply = {"mode": "clarification", "message": msg}
        history.append({"role": "assistant", "content": msg})
        return reply

    # --------------------------------------------------
    # 6. ML Prediction
    # --------------------------------------------------

    if has_symptoms:
        try:
            top3 = predict_top3(context_text)

            if top3:
                top_disease, top_confidence = top3[0]
                confidence = float(top_confidence)
                disease_name = top_disease.strip().replace("_", " ").title()

                other_diseases = [
                    r[0].strip().replace("_", " ").title()
                    for r in top3[1:]
                    if float(r[1]) > 0.05
                ]

                # HIGH confidence → direct prediction
                if confidence >= CONFIDENCE_THRESHOLD:
                    msg = (
                        f"AI predicts **{disease_name}** "
                        f"with {confidence * 100:.1f}% confidence.\n\n"
                        "⚠ This is a preliminary assessment. "
                        "Please consult a qualified doctor."
                    )
                    reply = {
                        "mode": "ml_prediction",
                        "disease": disease_name,
                        "confidence": confidence,
                        "message": msg
                    }
                    history.append({"role": "assistant", "content": msg})
                    session["state"] = "PREDICTED"
                    session["last_disease"] = disease_name
                    session["last_confidence"] = confidence
                    session["clarification_count"] = 0
                    return reply

                # MEDIUM confidence → ask clarifying questions first
                elif confidence >= LLM_THRESHOLD:
                    clarification_count = session.get("clarification_count", 0)

                    if clarification_count == 0:
                        msg = (
                            "I've detected some symptoms but need more "
                            "details to give you an accurate assessment.\n\n"
                            "Please answer the following:\n"
                            "- How long have you had these symptoms?\n"
                            "- How severe are they (mild/moderate/severe)?\n"
                            "- Do you have any fever? If yes, how high?\n"
                            "- Any recent travel, diet change, or exposures?\n"
                            "- Are symptoms getting better or worse?"
                        )
                        session["clarification_count"] = 1
                        session["pending_disease"] = disease_name
                        session["pending_confidence"] = confidence
                        session["pending_alternatives"] = other_diseases
                        session["state"] = "COLLECTING"

                    else:
                        pending_disease = session.get("pending_disease", disease_name)
                        pending_confidence = session.get("pending_confidence", confidence)
                        pending_alternatives = session.get("pending_alternatives", [])

                        alternatives = ""
                        if pending_alternatives:
                            alternatives = (
                                f" Other possibilities include "
                                f"{', '.join(pending_alternatives)}."
                            )

                        msg = (
                            f"Based on your symptoms and the details you "
                            f"provided, the closest match is "
                            f"**{pending_disease}** "
                            f"({pending_confidence * 100:.1f}% confidence), "
                            f"but this is not certain.{alternatives}\n\n"
                            "Please consult a qualified doctor for accurate "
                            "diagnosis.\n"
                            "⚠ This is a preliminary AI assessment only."
                        )
                        session["state"] = "PREDICTED"
                        session["last_disease"] = pending_disease
                        session["last_confidence"] = pending_confidence
                        session["clarification_count"] = 0
                        session.pop("pending_disease", None)
                        session.pop("pending_confidence", None)
                        session.pop("pending_alternatives", None)

                    reply = {"mode": "clarification", "message": msg}
                    history.append({"role": "assistant", "content": msg})
                    return reply

                # LOW confidence → Groq fallback
                else:
                    symptom_count = len(current_symptoms) + len(context_symptoms)

                    # Few symptoms + first attempt → ask clarifying questions
                    if (
                        symptom_count <= 2
                        and session.get("clarification_count", 0) == 0
                    ):
                        msg = (
                            "I've detected some symptoms but need more "
                            "details to give you an accurate assessment.\n\n"
                            "Please answer the following:\n"
                            "- How long have you had these symptoms?\n"
                            "- How severe are they (mild/moderate/severe)?\n"
                            "- Do you have any fever? If yes, how high?\n"
                            "- Any recent travel, diet change, or exposures?\n"
                            "- Are symptoms getting better or worse?"
                        )
                        session["clarification_count"] = 1
                        session["pending_disease"] = disease_name
                        session["pending_confidence"] = confidence
                        session["pending_alternatives"] = other_diseases
                        session["state"] = "COLLECTING"
                        reply = {"mode": "clarification", "message": msg}
                        history.append({"role": "assistant", "content": msg})
                        return reply

                    # Enough symptoms or already clarified → Groq
                    try:
                        groq_response = ask_groq_medical(
                            user_input, disease_name, confidence
                        )
                        if groq_response:
                            msg = (
                                "Our ML model wasn't confident enough for a "
                                "specific diagnosis from our trained diseases.\n\n"
                                "Here's what our AI health assistant "
                                f"suggests:\n\n{groq_response}\n\n"
                                "⚠ This is a preliminary assessment only. "
                                "Please consult a qualified doctor."
                            )
                            reply = {"mode": "llm_fallback", "message": msg}
                            history.append({"role": "assistant", "content": msg})
                            session["state"] = "COLLECTING"
                            return reply
                    except Exception:
                        pass

                    if confidence < MINIMUM_DISPLAY_CONFIDENCE:
                        msg = (
                            "I wasn't able to identify a specific condition "
                            "from your symptoms.\n\n"
                            "Please describe your symptoms in more detail:\n"
                            "- Which symptom is bothering you most?\n"
                            "- Any fever, rash, or difficulty breathing?\n"
                            "- How long have you been feeling this way?"
                        )
                    else:
                        msg = (
                            f"Your symptoms don't strongly match any single "
                            f"condition in our database (best guess: "
                            f"**{disease_name}** at "
                            f"{confidence * 100:.1f}% — too low to be reliable).\n\n"
                            "Please describe your symptoms in more detail:\n"
                            "- Which symptom is bothering you most?\n"
                            "- Any fever, rash, or difficulty breathing?\n"
                            "- How long have you been feeling this way?"
                        )
                    reply = {"mode": "clarification", "message": msg}
                    history.append({"role": "assistant", "content": msg})
                    session["state"] = "COLLECTING"
                    return reply

        except Exception as e:
            print("ML prediction error:", e)

    # --------------------------------------------------
    # 7. Insufficient Symptoms
    # --------------------------------------------------

    if any(word in text for word in [
        "feel", "sick", "unwell", "ill", "hurt", "ache", "pain",
        "symptom", "problem", "issue", "trouble", "concerned"
    ]):
        # Try Groq first before generic template
        try:
            groq_response = ask_groq_medical(user_input, "unknown", 0.0)
            if groq_response:
                msg = (
                    "Here's what our AI health assistant suggests based "
                    f"on what you've described:\n\n{groq_response}\n\n"
                    "⚠ This is a preliminary assessment only. "
                    "Please consult a qualified doctor."
                )
                reply = {"mode": "llm_fallback", "message": msg}
                history.append({"role": "assistant", "content": msg})
                session["state"] = "COLLECTING"
                return reply
        except Exception:
            pass

        msg = (
            "I can help analyze your symptoms. "
            "Please describe what you're experiencing in more detail.\n\n"
            "For example:\n"
            '- "I have fever, headache and body ache"\n'
            '- "Stomach pain with vomiting since 2 days"\n'
            '- "Cough, cold and sore throat"'
        )
        reply = {"mode": "clarification", "message": msg}
        history.append({"role": "assistant", "content": msg})
        session["state"] = "COLLECTING"
        return reply

    # --------------------------------------------------
    # 8. LLM Fallback (chitchat / fully OOD)
    # --------------------------------------------------

    try:
        reply_text = ask_ollama(user_input)
        if not reply_text:
            reply_text = (
                "I'm here to help with health questions. "
                "Could you describe any symptoms you're experiencing?"
            )
        reply = {"mode": "llm_fallback", "message": reply_text}
        history.append({"role": "assistant", "content": reply_text})
        return reply

    except Exception as e:
        print("LLM error:", e)
        msg = (
            "I can help you analyze health symptoms. "
            "Please describe your symptoms in detail, for example:\n\n"
            '- "I have fever, headache and body ache"\n'
            '- "Stomach pain with vomiting since 2 days"\n'
            '- "Cough, cold and sore throat"\n\n'
            "The more symptoms you share, the more accurate "
            "my prediction will be."
        )
        return {"mode": "clarification", "message": msg}