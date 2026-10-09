"""
Naive Bayes probabilistik untuk Sistem Pakar Diagnosis Penyakit.

Model: Bernoulli Naive Bayes (fitur = gejala biner: ada / tidak ada).
Diimplementasi manual (tanpa sklearn) supaya perhitungannya transparan
dan mudah dipertahankan di depan dosen.

    P(penyakit | gejala) = P(penyakit) * Π P(gejala_i | penyakit)
                          ─────────────────────────────────────────
                                           P(gejala)

Dihitung dalam ruang log lalu dinormalisasi (softmax) sehingga total
probabilitas semua penyakit = 100%.

SUMBER PELATIHAN (penting, dibaca dulu):
    File data/Testing-selected-columns.csv saat ini KOSONG (hanya header,
    tanpa baris data). Karena itu model ini "dilatih" dari basis aturan
    (RULES di app.py): tiap penyakit dianggap punya 1 contoh prototipe
    = himpunan premis rule-nya, lalu dihaluskan dengan Laplace smoothing
    (alpha=1):

        P(gejala | penyakit) = (cocok + alpha) / (1 + 2*alpha)

    - gejala premis     -> (1+1)/(1+2) = 2/3
    - gejala non-premis -> (0+1)/(1+2) = 1/3

    Prior P(penyakit) uniform = 1/4 (4 penyakit, masing-masing 1 rule).

    Begitu data latih asli tersedia (CSV terisi + kolom label penyakit),
    panggil train_from_csv() untuk mengestimasi ulang parameter dari data
    — tanpa mengubah kode inferensi.
"""

import csv
import math

# Disalin dari app.py agar modul ini bisa dipakai/diuji mandiri.
# (app.py mengimpor konstanta ini dari sini supaya satu sumber kebenaran.)
SYMPTOM_CODES = [
    "demam", "menggigil", "sakit_kepala",
    "nyeri_otot", "bintik_merah",
    "batuk", "pilek",
    "mual", "nyeri_perut",
]

DISEASES = ["Malaria", "Demam Berdarah", "Flu", "Tifus"]

# Premis tiap penyakit (disalin dari RULES di app.py).
PREMISES = {
    "Malaria":        {"demam", "menggigil", "sakit_kepala"},
    "Demam Berdarah": {"demam", "bintik_merah", "nyeri_otot"},
    "Flu":            {"batuk", "pilek", "demam"},
    "Tifus":          {"mual", "nyeri_perut", "demam"},
}

ALPHA = 1.0          # Laplace smoothing
_P_PREMIS = (1 + ALPHA) / (1 + 2 * ALPHA)   # = 2/3
_P_NON_PREMIS = ALPHA / (1 + 2 * ALPHA)      # = 1/3


def _likelihood(gejala: str, penyakit: str) -> float:
    """P(gejala=ada | penyakit), dari prototipe rule + smoothing."""
    if gejala in PREMISES[penyakit]:
        return _P_PREMIS
    return _P_NON_PREMIS


def predict_proba(gejala_terpilih):
    """Kembalikan [(penyakit, probabilitas), ...] terurut menurun.

    gejala_terpilih: iterable kode gejala, mis. ["demam", "batuk"].
    """
    dipilih = set(gejala_terpilih) & set(SYMPTOM_CODES)
    n_kelas = len(DISEASES)
    log_prior = math.log(1.0 / n_kelas)

    log_scores = {}
    for penyakit in DISEASES:
        s = log_prior
        for g in SYMPTOM_CODES:
            p = _likelihood(g, penyakit)
            s += math.log(p) if g in dipilih else math.log(1.0 - p)
        log_scores[penyakit] = s

    # Softmax: ubah skor log menjadi probabilitas (total = 1.0).
    m = max(log_scores.values())
    exps = {d: math.exp(v - m) for d, v in log_scores.items()}
    total = sum(exps.values())
    hasil = [(d, exps[d] / total) for d in DISEASES]
    hasil.sort(key=lambda x: -x[1])
    return hasil


def explain(penyakit, gejala_terpilih):
    """Kontribusi tiap gejala ke skor log penyakit (untuk debugging)."""
    dipilih = set(gejala_terpilih) & set(SYMPTOM_CODES)
    rincian = []
    for g in SYMPTOM_CODES:
        p = _likelihood(g, penyakit)
        kontribusi = math.log(p) if g in dipilih else math.log(1.0 - p)
        rincian.append((g, g in dipilih, round(kontribusi, 3)))
    return rincian


def train_from_csv(path, kolom_label="penyakit"):
    """Estimasi ulang P(gejala|penyakit) dari CSV data latih asli.

    Format CSV yang diharapkan: satu kolom label penyakit (nama kolom
    bisa diubah via kolom_label) + kolom-kolom gejala berisi 1/0
    (atau "ya"/"tidak", "true"/"false"). Baris kosong dilewati.

    Mengembalikan dict {(penyakit, gejala): probabilitas} yang bisa
    dipakai menggantikan _likelihood(). Saat ini CSV di data/ masih
    kosong sehingga fungsi ini belum dipanggil di app.py.
    """
    positif, total = {}, {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            penyakit = (row.get(kolom_label) or "").strip()
            if penyakit not in DISEASES:
                continue
            total[penyakit] = total.get(penyakit, 0) + 1
            for g in SYMPTOM_CODES:
                val = (row.get(g) or "").strip().lower()
                if val in ("1", "ya", "y", "true", "t"):
                    positif[(penyakit, g)] = positif.get((penyakit, g), 0) + 1

    estimasi = {}
    for penyakit in DISEASES:
        n = total.get(penyakit, 0)
        for g in SYMPTOM_CODES:
            c = positif.get((penyakit, g), 0)
            estimasi[(penyakit, g)] = (c + ALPHA) / (n + 2 * ALPHA)
    return estimasi


if __name__ == "__main__":
    for kasus in (
        ["demam", "menggigil", "sakit_kepala"],
        ["batuk", "pilek"],
        ["demam"],
        [],
    ):
        print("Gejala:", kasus or "-")
        for d, p in predict_proba(kasus):
            print(f"  {d:15} {p * 100:6.2f}%")
        print()
