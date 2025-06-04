import os
import faiss
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from keras.saving import register_keras_serializable # type: ignore

from config import (
    RETRIEVER_MODEL_PATH,
    RETRIEVER_TOKENIZER_PATH,
    FAISS_INDEX_PATH,
    KNOWLEDGE_BASE_DF_PATH,
    KNOWLEDGE_BASE_CSV_DIR,
    RETRIEVER_MAX_LEN,
    RETRIEVAL_THRESHOLD,
    VOCAB_SIZE,
    OOV_TOKEN
)

# Registrasi custom objects untuk Keras model
@register_keras_serializable(package="Custom")
def triplet_loss(y_true, y_pred, margin=0.5):
    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

@register_keras_serializable(package="Custom")
def l2_normalize(t):
    return tf.math.l2_normalize(t, axis=1)

# Variabel Global untuk Model dan Data
shared_encoder = None
retriever_keras_tokenizer = None
faiss_index = None
kbase_records = []

def initialize_retriever_resources():
    global shared_encoder, retriever_keras_tokenizer, faiss_index, kbase_records
    
    try:
        print("Flask API: Mencoba memuat model retriever Keras...")
        retriever_model_full = load_model(
            RETRIEVER_MODEL_PATH,
            custom_objects={
                "triplet_loss": triplet_loss,
                "tf": tf,
                "l2_normalize": l2_normalize
            },
            compile=False # Penting untuk inferensi
        )
        shared_encoder = retriever_model_full.get_layer("shared_encoder")
        print('Flask API: Model retriever (shared_encoder) berhasil dimuat.')

        if os.path.exists(RETRIEVER_TOKENIZER_PATH):
            with open(RETRIEVER_TOKENIZER_PATH, 'rb') as handle:
                retriever_keras_tokenizer = pickle.load(handle)
            print(f"Flask API: Tokenizer Keras untuk retriever dimuat dari {RETRIEVER_TOKENIZER_PATH}.")
        else:
            print(f"Flask API: Tokenizer Keras ({RETRIEVER_TOKENIZER_PATH}) tidak ditemukan, membuat baru...")
            # Logic membuat tokenizer seperti di FastAPI (atau notebook chatbot_pipeline)
            temp_df_for_tokenizer = []
            for filename in os.listdir(KNOWLEDGE_BASE_CSV_DIR):
                if filename.endswith(".csv"):
                    csv_path = os.path.join(KNOWLEDGE_BASE_CSV_DIR, filename)
                    df_kb_single = pd.read_csv(csv_path)
                    temp_df_for_tokenizer.append(df_kb_single)
            if not temp_df_for_tokenizer:
                 raise FileNotFoundError(f"Tidak ada file CSV di {KNOWLEDGE_BASE_CSV_DIR} untuk tokenizer.")
            combined_kb_df_for_tokenizer = pd.concat(temp_df_for_tokenizer, ignore_index=True)
            symptom_texts_for_tokenizer = combined_kb_df_for_tokenizer["problem_description"].astype(str).tolist()
            
            retriever_keras_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
            retriever_keras_tokenizer.fit_on_texts(symptom_texts_for_tokenizer)
            with open(RETRIEVER_TOKENIZER_PATH, 'wb') as handle:
                pickle.dump(retriever_keras_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Flask API: Tokenizer Keras untuk retriever disimpan ke {RETRIEVER_TOKENIZER_PATH}")


        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(KNOWLEDGE_BASE_DF_PATH):
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            kbase_df = pd.read_pickle(KNOWLEDGE_BASE_DF_PATH)
            print(f"Flask API: Indeks FAISS ({FAISS_INDEX_PATH}) dan KB DataFrame ({KNOWLEDGE_BASE_DF_PATH}) berhasil dimuat.")
        else:
            print(f"Flask API: FAISS index atau KB DataFrame tidak ditemukan, membuat baru...")
            all_data_kb = []
            for filename in os.listdir(KNOWLEDGE_BASE_CSV_DIR):
                if filename.endswith(".csv"):
                    file_path = os.path.join(KNOWLEDGE_BASE_CSV_DIR, filename)
                    df_kb_single = pd.read_csv(file_path)
                    all_data_kb.append(df_kb_single)
            if not all_data_kb:
                raise FileNotFoundError(f"Tidak ada file CSV di {KNOWLEDGE_BASE_CSV_DIR} untuk FAISS.")
            kbase_df = pd.concat(all_data_kb, ignore_index=True)
            kbase_df.to_pickle(KNOWLEDGE_BASE_DF_PATH)
            print(f"Flask API: KB DataFrame disimpan ke {KNOWLEDGE_BASE_DF_PATH}")

            symptom_texts = kbase_df["problem_description"].astype(str).tolist()
            kbase_vectors_list = []
            print("Flask API: Membuat embeddings untuk FAISS index...")
            for i, text_symptom in enumerate(symptom_texts):
                seq = retriever_keras_tokenizer.texts_to_sequences([text_symptom])
                pad_val = pad_sequences(seq, maxlen=RETRIEVER_MAX_LEN)
                embedding = shared_encoder.predict(pad_val, verbose=0)
                kbase_vectors_list.append(embedding[0])
                if (i + 1) % 200 == 0: # Kurangi frekuensi print
                    print(f"Flask API: Memproses embedding ke-{i+1}/{len(symptom_texts)}")
            
            if not kbase_vectors_list:
                raise ValueError("Tidak ada vektor yang dihasilkan untuk FAISS index.")

            kbase_vectors = np.array(kbase_vectors_list).astype("float32")
            faiss_index = faiss.IndexFlatL2(kbase_vectors.shape[1])
            faiss_index.add(kbase_vectors)
            faiss.write_index(faiss_index, FAISS_INDEX_PATH)
            print(f"Flask API: FAISS index disimpan ke {FAISS_INDEX_PATH}")

        kbase_records = kbase_df[["problem_description", "first_aid"]].rename(
            columns={"problem_description": "symptom"}
        ).to_dict(orient="records")
        print("Flask API: Inisialisasi modul retriever selesai.")

    except Exception as e:
        print(f"Flask API Error: Gagal besar saat inisialisasi modul retriever. {e}")
        shared_encoder = None
        retriever_keras_tokenizer = None
        faiss_index = None
        kbase_records = []

def embed_text_for_retrieval(text: str) -> np.ndarray:
    if not shared_encoder or not retriever_keras_tokenizer:
        raise ValueError("Retriever model atau Keras tokenizer tidak dimuat dengan benar.")
    seq = retriever_keras_tokenizer.texts_to_sequences([text])
    pad_val = pad_sequences(seq, maxlen=RETRIEVER_MAX_LEN)
    embedding = shared_encoder.predict(pad_val, verbose=0)
    return embedding[0]

def retrieve_solution(user_text: str) -> dict:
    if not faiss_index or not kbase_records: # Ditambahkan pengecekan kbase_records
        print("Flask API Warning: FAISS index atau knowledge base tidak tersedia.")
        return {"symptom": None, "first_aid": "Knowledge base tidak tersedia saat ini."}
    try:
        vec = embed_text_for_retrieval(user_text).astype("float32").reshape(1, -1)
        distances, indices = faiss_index.search(vec, 1)
        
        distance = distances[0][0]
        retrieved_idx = indices[0][0]
        
        # print(f"[DEBUG Flask] Jarak vektor: {distance}")
        
        if distance > RETRIEVAL_THRESHOLD or retrieved_idx >= len(kbase_records):
            return {"symptom": None, "first_aid": "Mohon maaf, kami tidak menemukan solusi yang cukup relevan. Bisa dijelaskan lebih rinci?"}
        
        return kbase_records[retrieved_idx]
    except Exception as e:
        print(f"Flask API Error saat retrieve_solution: {e}")
        return {"symptom": None, "first_aid": "Terjadi kesalahan saat mencari solusi."}

# Panggil inisialisasi saat modul ini diimpor pertama kali
initialize_retriever_resources()