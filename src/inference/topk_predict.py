import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models_saved/intent_respiratory5"
TOP_K = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Load labels
id2label = {}
with open(f"{MODEL_DIR}/labels.txt") as f:
    for line in f:
        i, l = line.strip().split("\t")
        id2label[int(i)] = l
labels = [id2label[i] for i in range(len(id2label))]

def predict(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    topk_idx = probs.argsort()[-TOP_K:][::-1]
    return [(labels[i], float(probs[i])) for i in topk_idx]

# Example
if __name__ == "__main__":
    text = "I have cough, chest tightness, and mild fever"
    preds = predict(text)
    for d, p in preds:
        print(f"{d}: {p:.2f}")
