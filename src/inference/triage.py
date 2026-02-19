import os, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.prompts import FOLLOW_UP, DEFAULT
from src.utils.safety_rules import wrap_safe

MODEL_DIR = os.getenv("INTENT_MODEL_DIR", "models_saved/intent")

class Triage:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        self.mdl.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mdl.to(self.device)
        self.id2label = self.mdl.config.id2label

    def classify(self, text: str):
        enc = self.tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = self.mdl(**enc).logits
        p = F.softmax(logits, dim=-1).cpu().numpy()[0]
        y = int(p.argmax())
        return self.id2label[y], float(p[y])

    def reply(self, user_text: str):
        label, conf = self.classify(user_text)
        fu = FOLLOW_UP.get(label, DEFAULT)
        tip_key = "dengue" if "dengue" in label.lower() else "general"
        base = f"It could be related to {label}. {fu}"
        return wrap_safe(base, condition_key=tip_key)

if __name__ == "__main__":
    t = Triage()
    print(t.reply("hey i feel high fever, severe headache, nausea, vomiting"))
