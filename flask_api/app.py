import os
# Set environment variables di awal, sebelum import TensorFlow/Keras
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Nonaktifkan oneDNN jika ada isu kompatibilitas
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Untuk mengatasi isu duplikasi library OpenMP jika muncul

from flask import Flask, request, jsonify
# Import modul-modul Anda. Ini akan memicu pemuatan model di dalamnya.
import intent_module
import retriever_module # Inisialisasi retriever (termasuk FAISS) akan dipanggil di sini
import chatbot_logic
from flask_cors import CORS
app = Flask(__name__)

CORS(app)

# Pemuatan model dan resource lain sudah terjadi saat modul diimpor.

@app.route('/chatbot-reply', methods=['POST'])
def reply_endpoint(): # Mengganti nama fungsi agar tidak bentrok dengan variabel 'reply'
    try:
        data = request.get_json()
        user_input = data.get('text') # Sesuai dengan Pydantic model di FastAPI ('text')
        latitude = data.get('latitude')
        longitude = data.get('longitude')

        if user_input is None: # Lebih baik cek None daripada not user_input untuk string kosong
            return jsonify({"error": "Input 'text' tidak ditemukan atau kosong"}), 400
        if not isinstance(user_input, str):
             return jsonify({"error": "Input 'text' harus berupa string"}), 400


        # 1. Prediksi Intent
        intent = intent_module.predict_intent(user_input)

        # 2. Dapatkan Balasan Chatbot
        response_message = chatbot_logic.chatbot_reply(user_input, intent, latitude, longitude) # Ganti nama variabel

        return jsonify({"reply": response_message}) # Sesuai output FastAPI

    except Exception as e:
        app.logger.error(f"Error pada endpoint /chatbot-reply: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Terjadi kesalahan internal pada server."}), 500

if __name__ == '__main__':
    # Port bisa disesuaikan. debug=True hanya untuk pengembangan.
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 8000)))