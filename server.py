"""
app.py — Backend Flask untuk Model Prediksi Risiko Kredit
=========================================================
Cara menjalankan:
  1. Pastikan Python & pip sudah terinstall
  2. Install dependencies:
       pip install flask flask-cors scikit-learn numpy
  3. Letakkan file ini di folder yang sama dengan:
       - model_naba.pkl
       - encoderss.pkl
       - prediksi_kredit.html  (opsional jika pakai mode serve)
  4. Jalankan:
       python app.py
  5. Buka browser: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='.')
CORS(app)  # Izinkan cross-origin request dari browser

# ─── Load Model & Encoder ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'model_naba.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'encoderss.pkl'), 'rb') as f:
    encoders = pickle.load(f)

print("✅ Model berhasil dimuat:", type(model).__name__)
print("✅ Encoder berhasil dimuat:", list(encoders.keys()))

# ─── Urutan fitur WAJIB sesuai model ─────────────────────────────
FEATURE_ORDER = [
    'usia',
    'status_pekerjaan',
    'lama_bekerja_tahun',
    'pendapatan_tahunan',
    'skor_kredit',
    'lama_riwayat_kredit_tahun',
    'aset_tabungan',
    'hutang_saat_ini',
    'gagal_bayar_tercatat',
    'tunggakan_2thn_terakhir',
    'catatan_negatif',
    'tipe_produk',
    'tujuan_pinjaman',
    'jumlah_pinjaman',
    'suku_bunga',
    'rasio_hutang_terhadap_pendapatan',
    'rasio_pinjaman_terhadap_pendapatan',
    'rasio_pembayaran_terhadap_pendapatan'
]

# ─── Route: Serve halaman HTML utama ─────────────────────────────
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'prediksi_kredit.html')

# ─── Route: Prediksi ─────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validasi semua field ada
        missing = [f for f in FEATURE_ORDER if f not in data]
        if missing:
            return jsonify({'error': f'Field tidak lengkap: {missing}'}), 400

        # Susun fitur sesuai urutan
        features = [float(data[f]) for f in FEATURE_ORDER]
        X = np.array(features).reshape(1, -1)

        # Prediksi
        prediction = int(model.predict(X)[0])
        probability = model.predict_proba(X)[0].tolist()

        # Label hasil
        label = "Disetujui" if prediction == 1 else "Ditolak"

        return jsonify({
            'prediction': prediction,
            'label': label,
            'probability': probability,
            'prob_disetujui': round(probability[1] * 100, 2),
            'prob_ditolak': round(probability[0] * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── Route: Health check ─────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': type(model).__name__})

# ─── Run ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀 Server berjalan di: http://localhost:5000")
    print("📊 Endpoint prediksi: POST http://localhost:5000/predict\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
