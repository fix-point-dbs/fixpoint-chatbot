import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import numpy as np
from config import INTENT_MODEL_PATH, INTENT_TOKENIZER_PATH, INTENT_LABELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
intent_model = None

try:
    # 1. Tokenizer tetap dari checkpoint asli
    tokenizer = AutoTokenizer.from_pretrained(INTENT_TOKENIZER_PATH)

    # 2. Ambil config dari tokenizer checkpoint
    config = AutoConfig.from_pretrained(INTENT_TOKENIZER_PATH)
    config.num_labels = len(INTENT_LABELS)

    # 3. Bangun arsitektur model berdasarkan config
    intent_model = AutoModelForSequenceClassification.from_config(config)

    # 4. Load bobot dari file .pt (hasil fine-tune)
    intent_model.load_state_dict(torch.load(INTENT_MODEL_PATH, map_location=device))

    classifier_weights = intent_model.classifier.weight.data
    print("Classifier weight mean:", classifier_weights.mean().item())
    print("Classifier weight std:", classifier_weights.std().item())

    # 5. Kirim ke device dan mode eval
    intent_model.to(device)
    intent_model.eval()

    print(f"Flask API: Model intent dan tokenizer berhasil dimuat di device '{device}'.")

except Exception as e:
    print(f"Flask API Error: Gagal memuat model intent atau tokenizer.")
    print(f"DETAIL ERROR: {e}")


def predict_intent(text: str) -> str:
    if not intent_model or not tokenizer:
        print("Flask API Fallback: Model intent atau tokenizer tidak dimuat.")
        return "fallback"

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = intent_model(**inputs)
        logits = outputs.logits
        intent_idx = torch.argmax(logits, dim=-1).item()

        if intent_idx >= len(INTENT_LABELS):
            print("Index out of range:", intent_idx)
            return "fallback"

        intent_label = INTENT_LABELS[intent_idx]
        print(f"Predicted intent for '{text}': {intent_label}")
        return intent_label
