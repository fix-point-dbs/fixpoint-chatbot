from retriever_module import retrieve_solution # (Pastikan path import ini benar)
import requests
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Tambahkan import untuk fungsi pemanggil API bengkel terdekat jika sudah ada
# from some_module import call_api_bengkel_terdekat

def chatbot_reply(user_input: str, intent: str, latitude: float = None, longitude: float = None) -> str:

    print("User Input: ", user_input)
    print("Intent: ", intent)
    print("Latitude: ", latitude)
    print("Longitude: ", longitude)

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

    elif intent == "cari_bengkel_terdekat":
        return call_api_bengkel_terdekat(user_input, latitude, longitude)

    elif intent == "ask_bengkel_by_lokasi":
        lokasi = cari_bengkel_berdasarkan_lokasi(user_input)
        if lokasi:
            return lokasi
        else:
            return "Mohon sebutkan lokasi atau nama daerah agar saya bisa membantu mencarikan bengkel terdekat."


    else:
        return "Terjadi kesalahan dalam sistem. Silakan coba lagi atau hubungi teknisi kami."

import requests

def call_api_bengkel_terdekat(user_input: str, latitude: str, longitude: str) -> str:
    if latitude is None or longitude is None:
        return "Lokasi Anda tidak terdeteksi. Mohon pastikan GPS atau layanan lokasi di perangkat Anda sudah aktif."
    try:
        url = f"https://backend-fixpoint.adza-zarif.my.id/services?lat={latitude}&lng={longitude}&status=approved"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        list_bengkel = response_data.get('data', [])
        if not list_bengkel:
            return "Maaf, saat ini tidak ditemukan bengkel di sekitar lokasi Anda."
        pesan_pembuka = "Berikut adalah beberapa bengkel terdekat dari lokasi Anda:<br><ol>"
        list_info_bengkel = []
        for i, bengkel in enumerate(list_bengkel, 1):
            nama = bengkel.get('bussiness_name', 'Nama tidak tersedia').strip()
            alamat = bengkel.get('address', 'Alamat tidak tersedia')
            jarak_meter = bengkel.get('distance', 0)
            # jarak_km = jarak_meter / 1000
            telepon = bengkel.get('alternative_phone', 'Nomor tidak tersedia')
            service_id = bengkel.get('id', '')
            detail_url = f"/service/detail/{service_id}" if service_id else "#"
            info_kalimat = f"<li><b>{i}. {nama}</b><br>Alamat: {alamat}<br>Jarak: {jarak_meter:.2f} km<br>Telepon: {telepon}<br><a href=\"{detail_url}\" class=\"btn-detail\" target=\"_blank\">Lihat Detail</a></li>"
            list_info_bengkel.append(info_kalimat)
        pesan_balasan = pesan_pembuka + "\n".join(list_info_bengkel) + "</ol>\n<style>.btn-detail { display: inline-block; padding: 6px 12px; margin-top: 4px; background: #1976d2; color: #fff !important; border-radius: 4px; text-decoration: none; font-size: 14px; } .btn-detail:hover { background: #1565c0; }</style>"
        return pesan_balasan
    except requests.exceptions.RequestException as e:
        print(f"Error saat request API: {e}")
        return "Terjadi gangguan saat mencoba mencari bengkel terdekat. Mohon coba lagi sesaat lagi."
    except Exception as e:
        print(f"Error tak terduga: {e}")
        return "Terjadi kesalahan pada sistem kami. Silakan coba kembali."



# Load model NER
tokenizer = AutoTokenizer.from_pretrained("cahya/bert-base-indonesian-ner")
model = AutoModelForTokenClassification.from_pretrained("cahya/bert-base-indonesian-ner")

# Build pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def cari_bengkel_berdasarkan_lokasi(text: str) -> str:
    entities = ner_pipeline(text)
    kota_ditemukan = None
    for ent in entities:
        if ent['entity_group'] in ['LOC', 'GPE']:
            kota_ditemukan = ent['word']
            print(f"[NER] Lokasi terdeteksi: {kota_ditemukan}")
            break
    if not kota_ditemukan:
        print("[NER] Tidak ditemukan entitas lokasi pada input.")
        return "Mohon maaf, saya tidak dapat menemukan nama kota atau daerah dari pesan Anda. Silakan coba lagi dengan menyebutkan nama kota."
    try:
        url = f"https://backend-fixpoint.adza-zarif.my.id/services?city={kota_ditemukan}&status=approved"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        list_bengkel = response_data.get('data', [])
        if not list_bengkel:
            return f"Maaf, tidak ditemukan bengkel yang terdaftar di daerah {kota_ditemukan}."
        pesan_pembuka = f"Berikut beberapa bengkel yang tersedia di daerah {kota_ditemukan}:<br><ol>"
        deskripsi_bengkel = []
        for i, bengkel in enumerate(list_bengkel, 1):
            nama = bengkel.get('bussiness_name', 'Nama tidak tersedia').strip()
            alamat = bengkel.get('address', 'Alamat tidak tersedia')
            telepon = bengkel.get('alternative_phone', 'Nomor tidak tersedia')
            service_id = bengkel.get('id', '')
            detail_url = f"/service/detail/{service_id}" if service_id else "#"
            deskripsi_bengkel.append(f"<li><b>{i}. {nama}</b><br>Alamat: {alamat}<br>Telepon: {telepon}<br><a href=\"{detail_url}\" class=\"btn-detail\" target=\"_blank\">Lihat Detail</a></li>")
        pesan_balasan = pesan_pembuka + "\n".join(deskripsi_bengkel) + "</ol>\n<style>.btn-detail { display: inline-block; padding: 6px 12px; margin-top: 4px; background: #1976d2; color: #fff !important; border-radius: 4px; text-decoration: none; font-size: 14px; } .btn-detail:hover { background: #1565c0; }</style>"
        return pesan_balasan
    except requests.exceptions.RequestException as e:
        print(f"Error saat request API: {e}")
        return "Terjadi gangguan saat mencoba mencari bengkel. Mohon coba lagi sesaat lagi."
    except Exception as e:
        print(f"Error tak terduga: {e}")
        return "Terjadi kesalahan pada sistem kami. Silakan coba kembali."


