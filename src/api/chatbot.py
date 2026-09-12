import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

MEDICAL_SYSTEM_PROMPT = """You are a clinical AI assistant trained to help with preliminary symptom assessment. You follow structured medical reasoning.

When a patient describes symptoms, you must:
1. Identify the most likely condition based on symptoms (differential diagnosis)
2. Mention 1-2 alternative conditions it could be
3. Identify any RED FLAG symptoms that need emergency attention
4. Give ONE clear actionable next step

Rules you must follow:
- Always be medically accurate and specific
- Never diagnose definitively — always say "suggests" or "may indicate"
- Always recommend consulting a qualified doctor
- If symptoms suggest emergency (chest pain + sweating, difficulty breathing, loss of consciousness) — say SEEK EMERGENCY CARE IMMEDIATELY
- Keep response under 150 words
- Never repeat the user's symptoms back to them
- Structure your response clearly with these sections:
  Assessment: [most likely condition]
  Could also be: [1-2 alternatives]
  Watch out for: [red flags if any]
  Next step: [what to do]"""

CHITCHAT_SYSTEM_PROMPT = """You are a friendly AI Health Assistant.
You help users understand their symptoms and guide them to describe what they are experiencing.
Keep responses short, warm and helpful.
Always gently guide the conversation back to health symptoms.
Never provide personal opinions on non-medical topics.
If asked what you can do, explain you analyze symptoms and suggest possible conditions."""


def _call_groq(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 200,
    temperature: float = 0.7
) -> str:
    if not GROQ_API_KEY:
        print("Groq: GROQ_API_KEY not set — check .env file")
        return ""

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=15
        )

        result = response.json()

        if "choices" not in result:
            print(f"Groq unexpected response: {result}")
            return ""

        return result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        print("Groq timeout — API took too long to respond")
        return ""
    except Exception as e:
        print(f"Groq error: {e}")
        return ""


def ask_ollama(user_input: str) -> str:
    """Groq chitchat and general health queries."""
    return _call_groq(
        system_prompt=CHITCHAT_SYSTEM_PROMPT,
        user_message=user_input,
        max_tokens=150,
        temperature=0.7
    )


def ask_groq_medical(symptoms: str, top_guess: str, confidence: float) -> str:
    """
    Structured medical assessment for conditions outside the 70 trained classes.
    Returns empty string if Groq is unavailable — main.py handles the fallback.
    """
    user_message = (
        f"Patient symptoms: {symptoms}\n\n"
        f"Note: Our ML classifier's best guess was '{top_guess}' "
        f"at {confidence * 100:.1f}% confidence — too low to be reliable. "
        f"Please provide your clinical assessment based purely on the symptoms."
    )
    return _call_groq(
        system_prompt=MEDICAL_SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=250,
        temperature=0.3
    )