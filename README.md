## Persiapan Awal (Penting!)

Model-model dan file-file yang telah di-training (seperti model ONNX, model Keras, tokenizer, indeks FAISS, dan DataFrame knowledge base yang sudah diproses) disimpan dalam folder `generated`. Karena ukuran file-file ini bisa sangat besar, mereka tidak disertakan langsung di repositori ini.

**Anda perlu mengunduh folder `generated` yang lengkap dari link Google Drive berikut dan meletakkannya di root direktori proyek Anda:**

➡️ **https://drive.google.com/file/d/1TSLaV2dG2WoSbbLMO5d9pANe_vqMNn4a/view?usp=sharing**

Pastikan setelah diunduh dan diekstrak (jika dalam format zip), struktur folder `generated` Anda sesuai dengan yang dibutuhkan oleh konfigurasi API (lihat `config.py` di dalam `flask_api/` atau path yang digunakan di `api/main.py`).

## Persyaratan Umum

* Python 3.10
* Pip (Python package installer)
* Virtual environment (dianjurkan)

## Metode yang Digunakan

Chatbot ini menggunakan pendekatan hybrid yang menggabungkan beberapa teknik Natural Language Processing (NLP) dan Machine Learning:

1.  **Klasifikasi Intent (Intent Classification)**:
    * **Model**: Menggunakan model berbasis Transformer, yaitu **IndoBERT** (`indobenchmark/indobert-base-p1`), yang telah di-fine-tune untuk tugas klasifikasi teks. Model ini dilatih untuk mengenali maksud atau tujuan dari input teks pengguna (misalnya, apakah pengguna bertanya tentang solusi, melaporkan masalah, atau hanya menyapa).
    * **Format**: Model yang telah di-fine-tune diekspor ke format **ONNX** (`intent_classifier.onnx`) untuk inferensi yang efisien.
    * **Tokenizer**: Menggunakan `AutoTokenizer` dari Hugging Face Transformers yang sesuai dengan IndoBERT (disimpan di `tokenizer_intent_classifier_indobert`).

2.  **Retrieval Informasi (Information Retrieval)**:
    * **Tujuan**: Untuk menemukan solusi atau informasi yang paling relevan dari *knowledge base* berdasarkan keluhan atau pertanyaan pengguna.
    * **Model Embedding**: Menggunakan arsitektur **Siamese BiLSTM** yang dilatih dengan *triplet loss*. Model ini menghasilkan *text embeddings* (representasi vektor dari teks) di mana teks dengan makna serupa akan memiliki vektor yang berdekatan dalam ruang embedding. Encoder dari model ini (`siamese_model.keras` atau `siamese_encoder_only.keras`) digunakan untuk menghasilkan embedding.
    * **Tokenizer**: Menggunakan `Tokenizer` dari Keras (`siamese_tokenizer.pkl`) yang telah di-fit pada korpus teks dari *knowledge base*.
    * **Pencarian Kemiripan**: **FAISS (Facebook AI Similarity Search)** digunakan untuk membangun indeks dari semua embedding entri *knowledge base*. Saat pengguna memberikan input, input tersebut diubah menjadi embedding, dan FAISS digunakan untuk secara efisien mencari embedding yang paling mirip (dengan jarak terdekat) di dalam indeks, sehingga menemukan entri *knowledge base* yang paling relevan.

3.  **Knowledge Base**:
    * Berisi pasangan antara deskripsi masalah/gejala (`problem_description`) dan solusi pertolongan pertama (`first_aid`) yang disimpan dalam file-file CSV di dalam folder `dataset/knowledge_base/`.

**Alur Kerja Umum Chatbot:**
1. Pengguna memberikan input teks.
2. Input teks diproses oleh model klasifikasi intent (IndoBERT ONNX) untuk menentukan maksud pengguna.
3. Berdasarkan intent yang terdeteksi:
    * Jika intent adalah sapaan, salam perpisahan, atau fallback, chatbot memberikan respons standar.
    * Jika intent adalah permintaan solusi, penyebab masalah, atau laporan gejala, input pengguna (atau bagian relevan darinya) diubah menjadi embedding menggunakan encoder Siamese BiLSTM.
    * Embedding ini kemudian digunakan untuk mencari entri yang paling mirip dalam indeks FAISS (yang berisi embedding dari *knowledge base*).
    * Solusi atau informasi yang relevan dari *knowledge base* kemudian diformat dan dikembalikan sebagai respons chatbot.

## Menjalankan API menggunakan FastAPI

API FastAPI berada di dalam folder `api/` dan file utamanya adalah `main.py`.

**Langkah-langkah:**

1.  **Buat dan Aktifkan Virtual Environment (Direkomendasikan):**
    ```bash
    python -m venv venv_fastapi
    # Windows
    venv_fastapi\Scripts\activate
    # macOS/Linux
    source venv_fastapi/bin/activate
    ```

2.  **Instal Dependensi:**
    Pastikan Anda memiliki file `requirements.txt` di dalam folder `api/` yang berisi semua pustaka yang dibutuhkan oleh FastAPI (misalnya, `fastapi`, `uvicorn`, `onnxruntime`, `transformers`, `tensorflow`, `keras`, `faiss-cpu`, `pandas`, `numpy`).
    ```bash
    cd api
    pip install -r requirements.txt 
    # (Jika requirements.txt belum ada, Anda perlu membuatnya berdasarkan import di main.py)
    ```
    Jika Anda menjalankan kode FastAPI yang telah dikirimkan sebelumnya, dependensi utamanya adalah:
    ```txt
    # Contoh isi requirements.txt untuk folder api/
    fastapi
    uvicorn[standard] # Untuk server ASGI
    numpy
    onnxruntime
    transformers
    pandas
    faiss-cpu # atau faiss-gpu jika menggunakan GPU
    tensorflow # atau tensorflow-cpu
    keras # Mungkin sudah termasuk dalam tensorflow
    pydantic
    python-dotenv # Jika Anda menggunakan .env untuk konfigurasi
    ```

3.  **Pindah ke Direktori API FastAPI:**
    Jika belum, pastikan Anda berada di dalam folder `api/`:
    ```bash
    cd path/ke/project_chatbot_motor/api/
    ```
    Jika Anda sudah di langkah sebelumnya, Anda mungkin sudah di sana.

4.  **Jalankan Server FastAPI menggunakan Uvicorn:**
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8001 
    ```
    * `main`: Merujuk ke file `main.py`.
    * `app`: Merujuk ke objek FastAPI `app = FastAPI()` di dalam `main.py`.
    * `--reload`: Server akan otomatis restart jika ada perubahan kode (berguna saat pengembangan).
    * `--host 0.0.0.0`: Membuat server dapat diakses dari alamat IP mana pun di mesin Anda.
    * `--port 8001`: Menentukan port yang akan digunakan (Anda bisa menggantinya).

5.  **Akses API:**
    * Endpoint chatbot Anda (sesuai kode FastAPI yang diberikan) adalah `POST` ke `http://127.0.0.1:8001/chatbot-reply/`

## Menjalankan API menggunakan Flask

API Flask berada di dalam folder `flask_api/` dan file utamanya adalah `app.py`.

**Langkah-langkah:**

1.  **Buat dan Aktifkan Virtual Environment (Direkomendasikan):**
    ```bash
    python -m venv venv_flask
    # Windows
    venv_flask\Scripts\activate
    # macOS/Linux
    source venv_flask/bin/activate
    ```

2.  **Instal Dependensi:**
    Pindah ke folder `flask_api/` dan instal dependensi dari `requirements.txt` yang ada di sana.
    ```bash
    cd path/ke/project_chatbot_motor/flask_api/
    pip install -r requirements.txt
    ```
    Contoh isi `requirements.txt` untuk `flask_api/`:
    ```txt
    # requirements.txt (di dalam flask_api/)
    flask
    onnxruntime
    transformers
    tensorflow # atau tensorflow-cpu
    keras # Mungkin sudah termasuk dalam tensorflow
    faiss-cpu # atau faiss-gpu
    pandas
    numpy
    python-dotenv # Jika Anda menggunakan .env untuk konfigurasi
    ```

3.  **Pindah ke Direktori API Flask:**
    Jika Anda belum berada di sana setelah instalasi dependensi:
    ```bash
    cd path/ke/project_chatbot_motor/flask_api/
    ```

4.  **Jalankan Aplikasi Flask:**
    ```bash
    python app.py
    ```
    Ini akan memulai server pengembangan Flask bawaan.

5.  **Proses Inisialisasi Retriever (Saat Pertama Kali Dijalankan untuk Flask):**
    Saat API Flask pertama kali dijalankan (atau jika file `faiss_index.idx`, `knowledge_base_df.pkl`, dan `siamese_tokenizer.pkl` di folder `generated` dihapus/belum ada), skrip akan mencoba membuat file-file ini. Proses ini mungkin memakan waktu beberapa menit. Untuk eksekusi selanjutnya, API akan memuat file yang sudah ada.

6.  **Akses API:**
    * Server Flask akan berjalan (default) di `http://127.0.0.1:8000` (atau port yang Anda tentukan di `app.run(...)`).
    * Endpoint chatbot Anda adalah `POST` ke `http://127.0.0.1:8000/chatbot-reply`

## Menguji API

Anda dapat menguji kedua API menggunakan tools seperti Postman, Insomnia, atau `curl`.

* **Metode**: `POST`
* **URL**:
    * FastAPI: `http://<host>:<port_fastapi>/chatbot-reply/`
    * Flask: `http://<host>:<port_flask>/chatbot-reply`
* **Headers**:
    * `Content-Type`: `application/json`
* **Body** (raw JSON):
    ```json
    {
        "text": "motor saya akinya soak dan tidak bisa menyala"
    }
    ```

Ganti `"motor saya akinya soak dan tidak bisa menyala"` dengan input teks yang ingin Anda uji.

## Catatan Tambahan

* Pastikan semua path ke model dan dataset di file konfigurasi (`config.py` untuk Flask, atau langsung di `main.py` untuk FastAPI) sudah benar dan sesuai dengan struktur direktori Anda setelah mengunduh folder `generated`.
* Untuk penggunaan produksi, disarankan menggunakan server WSGI yang lebih robust untuk Flask (seperti Gunicorn atau Waitress) dan server ASGI yang sesuai untuk FastAPI (Uvicorn sudah baik).
