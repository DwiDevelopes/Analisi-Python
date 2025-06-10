# Analisi-Python
<img src = "analis.jpg" width="100%" height= "100%">
**Analisi-Python** adalah proyek Python yang berfokus pada analisis data berbasis tanggal. Proyek ini dirancang untuk membantu pengguna dalam memproses, memvisualisasikan, dan menganalisis data waktu (time-series) menggunakan pustaka Python populer seperti `pandas`, `datetime`, dan `matplotlib`.

## 📌 Fitur Utama

- Ekstraksi informasi dari kolom tanggal (hari, bulan, tahun, kuartal, dll.)
- Konversi dan parsing format tanggal
- Filter dan agregasi data berdasarkan waktu
- Visualisasi tren berdasarkan waktu (time-series plot)
- Deteksi hari libur dan akhir pekan
- Penghitungan selisih waktu (duration)

## 📁 Struktur Folder

```

Analisi-Python/
│
├── data/               # Folder untuk dataset mentah (CSV/Excel)
├── notebooks/          # Jupyter Notebook untuk eksplorasi dan analisis
├── src/                # Kode sumber utama (modular)
│   └── date\_utils.py   # Fungsi-fungsi bantu untuk manipulasi tanggal
├── tests/              # Unit test
├── requirements.txt    # Daftar dependensi
└── README.md           # Dokumentasi proyek ini

````

## 📊 Contoh Analisis

Contoh kode untuk mengekstrak informasi tanggal dari kolom `tanggal` dalam DataFrame:

```python
import pandas as pd
from src.date_utils import extract_date_features

df = pd.read_csv("data/penjualan.csv")
df = extract_date_features(df, 'tanggal')

print(df[['tanggal', 'tahun', 'bulan', 'hari', 'hari_nama']].head())
````

Contoh output:

```
     tanggal  tahun  bulan  hari hari_nama
0 2024-01-01   2024      1     1    Senin
1 2024-01-02   2024      1     2    Selasa
...
```

## 🔧 Instalasi

1. **Clone repo ini**

   ```bash
   git clone https://github.com/username/Analisi-Python.git
   cd Analisi-Python
   ```

2. **Buat environment dan install dependensi**

   ```bash
   python -m venv env
   source env/bin/activate  # atau env\Scripts\activate di Windows
   pip install -r requirements.txt
   ```

## 🧪 Testing

Gunakan `pytest` untuk menjalankan unit test:

```bash
pytest tests/
```

## 📚 Dependensi

* Python >= 3.8
* pandas
* matplotlib
* seaborn
* numpy
* holidays
* jupyter

Instal semua dependensi dengan:

```bash
pip install -r requirements.txt
```

## ✅ Kontribusi

Pull request sangat disambut! Untuk perubahan besar, silakan buka issue terlebih dahulu agar kita bisa berdiskusi terlebih dahulu.

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

## 📬 Kontak

Dibuat oleh \[Dwi Bakti N Dev]

```
