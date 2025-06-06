from retriever_module import retrieve_solution # (Pastikan path import ini benar)

def chatbot_reply(user_input: str, intent: str) -> str:
    # Jika intent membutuhkan pencarian solusi di knowledge base
    if intent in ["ask_first_aid_solution", "ask_possible_cause", "report_noise_or_smell"]:
        result = retrieve_solution(user_input)
        symptom = result.get('symptom')
        first_aid = result.get('first_aid')

        # Cek apakah retriever berhasil menemukan gejala yang relevan
        if symptom:
            if intent == "ask_first_aid_solution":
                return f"Berdasarkan keluhan Anda, tindakan pertolongan pertama yang dapat dicoba adalah: {first_aid}"
            elif intent == "ask_possible_cause":
                # Menggabungkan gejala dan solusi menjadi satu paragraf yang informatif
                return f"Berdasarkan deskripsi Anda, kemungkinan masalahnya adalah '{symptom}'. Sebagai langkah awal, {first_aid[0].lower() + first_aid[1:]}"
            elif intent == "report_noise_or_smell":
                return f"Terima kasih atas laporannya. Indikasi awal dari keluhan tersebut bisa jadi adalah '{symptom}'. Kami sarankan untuk melakukan langkah berikut: {first_aid}"
        else:
            # Jika retriever tidak menemukan solusi, 'first_aid' akan berisi pesan fallback-nya
            # Contoh: "Mohon maaf, kami tidak menemukan solusi yang cukup relevan..."
            return first_aid

    # Blok untuk intent yang tidak memerlukan retriever
    elif intent == "casual_greeting":
        return "Halo! Ada keluhan atau masalah pada kendaraan Anda yang bisa saya bantu?"

    elif intent == "goodbye":
        return "Terima kasih! Semoga kendaraan Anda segera dalam kondisi baik."

    elif intent == "fallback":
        return "Maaf, saya belum memahami keluhan Anda. Bisa dijelaskan lebih lanjut?"

    else:
        return "Terjadi kesalahan dalam sistem. Silakan coba lagi atau hubungi teknisi kami."