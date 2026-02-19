DISCLAIMER = "This is general health information, not medical advice. For emergencies, seek immediate care."
EMERGENCY = ["severe chest pain", "trouble breathing", "self-harm", "uncontrolled bleeding"]

SAFE_TIPS = {
    "general": [
        "Rest and stay hydrated unless told otherwise by a clinician.",
        "Monitor temperature and note warning signs like persistent vomiting or bleeding.",
        "Avoid over‑the‑counter painkillers unless a clinician advised them."
    ],
    "dengue": [
        "Stay hydrated with oral fluids; watch for abdominal pain, bleeding, or persistent vomiting.",
        "Avoid aspirin or ibuprofen unless a clinician explicitly advised them.",
        "Seek prompt medical care if symptoms worsen or new warning signs appear."
    ]
}

def needs_escalation(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in EMERGENCY)

def wrap_safe(response: str, condition_key: str | None = None) -> str:
    key = condition_key or "general"
    tips = SAFE_TIPS.get(key, SAFE_TIPS["general"])
    bullets = "\n- ".join(tips)
    return f"{response}\n\nSuggestions:\n- {bullets}\n\n{DISCLAIMER}"
