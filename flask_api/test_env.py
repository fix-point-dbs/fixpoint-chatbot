# file: test_env.py
from transformers import BertTokenizer

# Ganti dengan nama model yang Anda gunakan untuk training
# Jika Anda tidak yakin, coba satu per satu antara 'base' dan 'small'
MODEL_NAME = "indobenchmark/indobert-lite-base-p1" 

print(f"Mencoba memuat tokenizer untuk: {MODEL_NAME}")
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    print("✅ BERHASIL: Tokenizer berhasil dimuat!")
    print(tokenizer)
except Exception as e:
    print("❌ GAGAL: Terbukti environment bermasalah.")
    print(f"DETAIL ERROR: {e}")