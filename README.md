# FixPoint : Chatbot Pertolongan Pertama Kerusakan Motor

Dokumentasi ini menjelaskan arsitektur, metode yang digunakan, dan cara menjalankan API untuk chatbot pertolongan pertama kerusakan motor.

---

### **Penting: Diperlukan Git LFS untuk Mengunduh Proyek**

Repositori ini berisi file model Machine Learning berukuran besar (seperti `.onnx` dan `.keras`). Untuk mengelola file-file ini dengan benar, proyek ini menggunakan **Git LFS (Large File Storage)**. Anda **wajib** menginstal dan mengaktifkan Git LFS **sebelum** melakukan `git clone`.

**Langkah-langkah yang Benar untuk Mengunduh:**

1.  **Instal Git LFS.** Kunjungi [situs resmi Git LFS](https://git-lfs.github.com/) untuk panduan instalasi sesuai sistem operasi Anda (Windows/macOS/Linux).

2.  **Jalankan setup Git LFS pada sistem Anda.** Perintah ini hanya perlu dijalankan sekali per mesin.
    ```bash
    git lfs install
    ```

3.  **Clone repositori seperti biasa.** Setelah Git LFS terinstal dan di-setup, proses `clone` akan secara otomatis mengunduh file-file besar dengan benar.
    ```bash
    git clone <URL_REPOSITORI_ANDA>
    ```

> **Catatan**: Jika Anda sudah terlanjur melakukan `clone` tanpa Git LFS, file model hanya akan berupa *pointer* teks kecil, bukan file sebenarnya. Untuk memperbaikinya, masuk ke direktori proyek dan jalankan `git lfs pull`.

---

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
    * Flask: `http://<host>:<port_flask>/chatbot-reply`
    * Deploy : `https://flask.fixpoint.my.id/chatbot-reply`
* **Headers**:
    * `Content-Type`: `application/json`
#### **Struktur Body (raw JSON)**

Endpoint ini menerima dua format *body* JSON yang berbeda, tergantung pada jenis permintaan yang Anda buat:

---

**Opsi 1: Permintaan Teks Umum**

Gunakan format ini untuk percakapan umum atau pertanyaan yang tidak memerlukan informasi lokasi.

* **Tujuan**: Mendapatkan balasan chatbot untuk pertanyaan umum.
* **Contoh Body**:
    ```json
    {
        "text": "motor saya akinya soak dan tidak bisa menyala"
    }
    ```
    Anda dapat mengganti nilai `"text"` dengan input lain yang relevan.

---

**Opsi 2: Permintaan Pencarian Berbasis Lokasi**

Gunakan format ini ketika Anda ingin mencari rekomendasi berbasis lokasi, seperti "bengkel terdekat". API akan menggunakan `latitude` dan `longitude` untuk memberikan saran yang relevan dengan lokasi Anda saat ini.

* **Tujuan**: Mendapatkan rekomendasi bengkel atau layanan lain berdasarkan lokasi GPS pengguna.
* **Contoh Body**:
    ```json
    {
        "text": "carikan bengkel terdekat",
        "latitude": "-8.163114",
        "longitude": "113.706995"
    }
    ```
    Pastikan untuk mengganti nilai `text`, `latitude`, dan `longitude` sesuai dengan kebutuhan pengujian Anda.

---

## Link Deploy
https://flask.fixpoint.my.id/chatbot-reply

## Catatan Tambahan

* Pastikan semua path ke model dan dataset di file konfigurasi `config.py` untuk Flask sudah benar dan sesuai dengan struktur direktori Anda setelah mengunduh folder `generated`.
