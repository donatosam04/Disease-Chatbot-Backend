import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def ask_ollama(user_input: str):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "phi3",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional AI health assistant. "
                        "Respond with ONE clear helpful reply only. "
                        "Do NOT simulate conversations. "
                        "Do NOT write User or Assistant labels. "
                        "Keep response natural and human-like. "
                        "If symptoms sound serious, advise medical attention."
                    )
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            "stream": False
        }
    )

    result = response.json()
    return result["message"]["content"].strip()
