import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import sys
import faiss
import tensorflow as tf
from keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# FastAPI app initialization
app = FastAPI()
print(os.listdir('./dataset'))

# Predict Intent
# Load the model
try :
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    print('Tokenizer Found.')
except Exception as e:
    print(f"Terjadi kesalahan saat load tokenizer. {e}")
    
try :
    onnx_path = "./generated/intent_classifier.onnx"
    intent_session = ort.InferenceSession(onnx_path)
except Exception as e:
    print(f'Terjadi kesalahan saat load onnx model. {e}')

# Intent labels (should match your model's labels)
intent_labels = [
    "ask_first_aid_solution",
    "ask_possible_cause",
    "casual_greeting",
    "fallback",
    "goodbye",
    "report_noise_or_smell"
]

def predict_intent(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=128
    )
    inputs_onnx = {k: v.cpu().numpy() for k, v in inputs.items()}
    
    input_ids_name = intent_session.get_inputs()[0].name
    attention_mask_name = intent_session.get_inputs()[1].name
    output_name = intent_session.get_outputs()[0].name

    outputs = intent_session.run([output_name], {
        input_ids_name: inputs_onnx["input_ids"],
        attention_mask_name: inputs_onnx["attention_mask"]
    })
    
    intent_idx = np.argmax(outputs[0], axis=1)[0]
    return intent_labels[intent_idx]


class TextInput(BaseModel):
    text: str

# Load Retriever
@register_keras_serializable(package="Custom")
def triplet_loss(y_true, y_pred, margin=0.5):
    
    # Bagi menjadi tiga embedding sepanjang dim-1
    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)

    # Hitung squared Euclidean distance
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    # Triplet loss
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

@register_keras_serializable(package="Custom")
def l2_normalize(t):
    return tf.math.l2_normalize(t, axis=1)

# Load model Siamese BiLSTM retriever
try :
    retriever_model = tf.keras.models.load_model(
        "./generated/siamese_model.keras",
        custom_objects={
            "triplet_loss": triplet_loss,
            "tf": tf,  # tambahkan ini
            "l2_normalize": l2_normalize
        }
    )
    print('Retriever model available.')
except Exception as e:
    print(f'Terjadi kesalahan saat load retriever. {e}')

# Load seluruh problem_description dari knowledge base CSV
folder_path = "./dataset/knowledge_base/"
all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder_path, filename))
        all_data.append(df)

knowledge_base_df = pd.concat(all_data, ignore_index=True)

# Ambil semua teks symptom/problem_description
symptom_texts = knowledge_base_df["problem_description"].astype(str).tolist()

# Fit tokenizer dengan semua kalimat pada knowledge base
VOCAB_SIZE = 10000
OOV_TOKEN = "<OOV>"
retriver_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
retriver_tokenizer.fit_on_texts(symptom_texts)

def embed_text(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=30)
    return retriever_model.predict(pad)[0]

encoder = retriever_model.get_layer("shared_encoder")
type(encoder)

# Encoder dari retriever model
encoder = retriever_model.get_layer("shared_encoder")  # Sudah valid

# Tokenizer harus disiapkan sebelumnya: retriver_tokenizer

# Fungsi embedding teks
def embed_text(text):
    seq = retriver_tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=30)  # Sesuaikan maxlen dengan model retriever
    embedding = encoder.predict(pad)
    return embedding[0]

# Baca semua file CSV di folder knowledge_base
folder_path = "./dataset/knowledge_base/"
all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        all_data.append(df)

# Gabungkan semua data
knowledge_base_df = pd.concat(all_data, ignore_index=True)

# Ubah menjadi list of dict
kbase = knowledge_base_df[["problem_description", "first_aid"]].rename(
    columns={"problem_description": "symptom"}
).to_dict(orient="records")

# Embedding semua data
kbase_vectors = np.array([embed_text(k["symptom"]) for k in kbase]).astype("float32")

# Bangun index FAISS
index = faiss.IndexFlatL2(kbase_vectors.shape[1])
index.add(kbase_vectors)

# Fungsi untuk retrieve solusi berdasarkan input user
RETRIEVAL_THRESHOLD = 0.4  # Atur eksperimen

def retrieve_solution(user_text):
    vec = embed_text(user_text).astype("float32").reshape(1, -1)
    D, I = index.search(vec, 1)
    distance = D[0][0]
    result = kbase[I[0][0]]
    
    print(f"[DEBUG] Jarak vektor: {distance}")
    
    if distance > RETRIEVAL_THRESHOLD:
        return {"symptom": None, "first_aid": "Mohon maaf, kami tidak menemukan solusi yang cukup relevan untuk masalah ini. Bisa dijelaskan lebih rinci?"}
    
    return result

def chatbot_reply(user_input):
    intent = predict_intent(user_input)

    if intent == "ask_first_aid_solution":
        result = retrieve_solution(user_input)
        return f"Pertolongan Pertama yang dapat Anda lakukan: {result['first_aid']}"
    
    elif intent == "ask_possible_cause":
        result = retrieve_solution(user_input)
        return f"Beberapa kemungkinan penyebabnya: {result}"

    elif intent == "report_noise_or_smell":
        result = retrieve_solution(user_input)
        return f"Terima kasih atas laporannya. Kami mendeteksi indikasi: {result}"

    elif intent == "casual_greeting":
        return "Halo! Ada keluhan atau masalah pada kendaraan Anda yang bisa saya bantu?"

    elif intent == "goodbye":
        return "Terima kasih! Semoga kendaraan Anda segera dalam kondisi baik."

    elif intent == "fallback":
        return "Maaf, saya belum memahami keluhan Anda. Bisa dijelaskan lebih lanjut?"

    else:
        return "Terjadi kesalahan dalam sistem. Silakan coba lagi atau hubungi teknisi kami."

@app.post("/chatbot-reply/")
async def reply(input_data: TextInput):
    print(input_data)
    reply = chatbot_reply(input_data.text)
    return {"reply": reply}

