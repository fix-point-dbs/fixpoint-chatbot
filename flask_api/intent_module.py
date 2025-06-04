import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from config import INTENT_MODEL_PATH, INTENT_TOKENIZER_PATH, INTENT_LABELS

# Muat sekali saat aplikasi dimulai
try:
    # Menggunakan path tokenizer yang telah di-fine-tune dari config.py
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    intent_session = ort.InferenceSession(INTENT_MODEL_PATH)
    input_ids_name = intent_session.get_inputs()[0].name
    attention_mask_name = intent_session.get_inputs()[1].name
    output_name = intent_session.get_outputs()[0].name
    print("Flask API: Model klasifikasi intent dan tokenizer (IndoBERT) berhasil dimuat.")
except Exception as e:
    print(f"Flask API Error: Gagal memuat model intent atau tokenizer. {e}")
    tokenizer = None
    intent_session = None

def predict_intent(text: str) -> str:
    if not intent_session or not tokenizer:
        print("Flask API Fallback: Model intent atau tokenizer tidak dimuat.")
        return "fallback"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=128
    )
    inputs_onnx = {k: v.cpu().numpy() for k, v in inputs.items()}

    outputs = intent_session.run([output_name], {
        input_ids_name: inputs_onnx["input_ids"],
        attention_mask_name: inputs_onnx["attention_mask"]
    })
    
    intent_idx = np.argmax(outputs[0], axis=1)[0]
    return INTENT_LABELS[intent_idx]