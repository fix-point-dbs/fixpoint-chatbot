from retriever_module import retrieve_solution # (Pastikan path import ini benar)

def chatbot_reply(user_input: str, intent: str) -> str:
    if intent == "ask_first_aid_solution":
        result = retrieve_solution(user_input)
        return f"Pertolongan Pertama yang dapat Anda lakukan: {result.get('first_aid', 'Tidak ada solusi spesifik saat ini.')}"
    
    elif intent == "ask_possible_cause":
        result = retrieve_solution(user_input)
        symptom = result.get('symptom', 'tidak teridentifikasi dengan jelas')
        first_aid = result.get('first_aid', 'tidak ada saran spesifik saat ini.')
        # Mengikuti format respons dari kode FastAPI Anda
        return f"Beberapa kemungkinan penyebabnya: {result}"


    elif intent == "report_noise_or_smell":
        result = retrieve_solution(user_input)
         # Mengikuti format respons dari kode FastAPI Anda
        return f"Terima kasih atas laporannya. Kami mendeteksi indikasi: {result}"

    elif intent == "casual_greeting":
        return "Halo! Ada keluhan atau masalah pada kendaraan Anda yang bisa saya bantu?"

    elif intent == "goodbye":
        return "Terima kasih! Semoga kendaraan Anda segera dalam kondisi baik."

    elif intent == "fallback":
        return "Maaf, saya belum memahami keluhan Anda. Bisa dijelaskan lebih lanjut?"

    else:
        return "Terjadi kesalahan dalam sistem. Silakan coba lagi atau hubungi teknisi kami."