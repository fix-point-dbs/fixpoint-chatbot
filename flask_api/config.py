import os

# Mendapatkan direktori tempat config.py berada (yaitu chatbot_flask_api)
BASE_APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Naik satu level untuk mencapai root project tempat folder 'generated' dan 'dataset' berada
PROJECT_ROOT_DIR = os.path.dirname(BASE_APP_DIR)

# Path Model & Data AKAN MERUJUK KE FOLDER ../generated/ dan ../dataset/
INTENT_MODEL_PATH = os.path.join(PROJECT_ROOT_DIR, "generated", "intent_classifier.onnx")
INTENT_TOKENIZER_PATH = os.path.join(PROJECT_ROOT_DIR, "generated", "tokenizer_intent_classifier_indobert") # Ini adalah folder
RETRIEVER_MODEL_PATH = os.path.join(PROJECT_ROOT_DIR, "generated", "siamese_model.keras")
RETRIEVER_TOKENIZER_PATH = os.path.join(PROJECT_ROOT_DIR, "generated", "siamese_tokenizer.pkl")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT_DIR, "generated", "faiss_index.idx")
KNOWLEDGE_BASE_DF_PATH = os.path.join(PROJECT_ROOT_DIR, "generated", "knowledge_base_df.pkl")
KNOWLEDGE_BASE_CSV_DIR = os.path.join(PROJECT_ROOT_DIR, "dataset", "knowledge_base")

# Label Intent
INTENT_LABELS = [
    "ask_first_aid_solution",
    "ask_possible_cause",
    "casual_greeting",
    "fallback",
    "goodbye",
    "report_noise_or_smell"
]

# Parameter Retriever
RETRIEVER_MAX_LEN = 30
RETRIEVAL_THRESHOLD = 0.4
VOCAB_SIZE = 10000
OOV_TOKEN = "<OOV>"