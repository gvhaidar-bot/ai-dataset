from flask import Flask, render_template, request

from nb_model import predict_proba as nb_predict_proba

app = Flask(__name__)

# ============================================================
# KNOWLEDGE BASE
# ============================================================
SYMPTOMS = [
    {"code": "demam",        "label": "Demam",        "icon": "🌡️"},
    {"code": "menggigil",    "label": "Menggigil",    "icon": "🥶"},
    {"code": "sakit_kepala", "label": "Sakit Kepala", "icon": "🤕"},
    {"code": "nyeri_otot",   "label": "Nyeri Otot",   "icon": "💪"},
    {"code": "bintik_merah", "label": "Bintik Merah", "icon": "🔴"},
    {"code": "batuk",        "label": "Batuk",        "icon": "😷"},
    {"code": "pilek",        "label": "Pilek",        "icon": "🤧"},
    {"code": "mual",         "label": "Mual",         "icon": "🤢"},
    {"code": "nyeri_perut",  "label": "Nyeri Perut",  "icon": "😖"},
]

RULES = [
    {"id": "R1", "premises": ["demam", "menggigil", "sakit_kepala"], "conclusion": "Malaria"},
    {"id": "R2", "premises": ["demam", "bintik_merah", "nyeri_otot"], "conclusion": "Demam Berdarah"},
    {"id": "R3", "premises": ["batuk", "pilek", "demam"],             "conclusion": "Flu"},
    {"id": "R4", "premises": ["mual", "nyeri_perut", "demam"],        "conclusion": "Tifus"},
]

DISEASES = ["Malaria", "Demam Berdarah", "Flu", "Tifus"]

SYMPTOM_LABEL = {s["code"]: s["label"] for s in SYMPTOMS}
SYMPTOM_ICON  = {s["code"]: s["icon"]  for s in SYMPTOMS}


# ============================================================
# SCORING  (Partial Match + Specificity)
# ============================================================
def _symptom_specificity(code):
    """
    Gejala yang muncul di banyak rule = gejala umum (skor rendah).
    Gejala yang cuma muncul di 1 rule = spesifik (skor tinggi).
    """
    count = sum(1 for r in RULES if code in r["premises"])
    return 1.0 / count if count else 1.0


def _score_rule(rule, fakta):
    """
    Skor = 60% rasio kecocokan + 40% bobot spesifisitas gejala.
    Return None kalau tidak ada satupun premis yang match.
    """
    premises = rule["premises"]
    matched  = [p for p in premises if p in fakta]

    # ← ini kunci: minimal 1 gejala match sudah cukup untuk muncul
    if not matched:
        return None

    partial = len(matched) / len(premises)

    total_spec   = sum(_symptom_specificity(p) for p in premises)
    matched_spec = sum(_symptom_specificity(p) for p in matched)
    specificity  = matched_spec / total_spec if total_spec else 0.0

    score = 0.6 * partial + 0.4 * specificity

    return {
        "rule_id":     rule["id"],
        "disease":     rule["conclusion"],
        "matched":     matched,
        "missing":     [p for p in premises if p not in fakta],
        "premises":    premises,
        "score":       score,
        "partial":     partial,
        "specificity": specificity,
    }


def _status(score):
    if score >= 0.85: return "Confirmed"
    if score >= 0.60: return "Likely"
    if score >= 0.40: return "Possible"
    return "Weak"


# ============================================================
# FORWARD CHAINING  (Data-Driven)
# ============================================================
def forward_chaining(fakta_gejala):
    fakta = set(fakta_gejala)
    results = []

    for rule in RULES:                     # ← iterasi dari sisi RULE
        info = _score_rule(rule, fakta)
        if info:
            info["status"] = _status(info["score"])
            results.append(info)

    results.sort(key=lambda x: -x["score"])
    return results


# ============================================================
# BACKWARD CHAINING  (Goal-Driven)
# ============================================================
def backward_chaining(fakta_gejala):
    fakta = set(fakta_gejala)
    results = []

    for goal in DISEASES:                  # ← iterasi dari sisi GOAL
        matching = [r for r in RULES if r["conclusion"] == goal]
        best = None
        for rule in matching:
            info = _score_rule(rule, fakta)
            if info and (best is None or info["score"] > best["score"]):
                best = info
        if best:
            best["goal"]   = goal
            best["status"] = _status(best["score"])
            results.append(best)

    results.sort(key=lambda x: -x["score"])
    return results


# ============================================================
# ROUTE  (sama seperti sebelumnya)
# ============================================================
@app.route("/", methods=["GET", "POST"])
def home():
    gejala_terpilih = []
    hasil_fc = []
    hasil_bc = []
    hasil_nb = []
    submitted = False

    if request.method == "POST":
        submitted = True
        gejala_terpilih = request.form.getlist("gejala")
        if gejala_terpilih:
            hasil_fc = forward_chaining(gejala_terpilih)
            hasil_bc = backward_chaining(gejala_terpilih)
            # Naive Bayes: probabilitas tiap penyakit (total 100%).
            hasil_nb = [
                {"disease": d, "probability": p}
                for d, p in nb_predict_proba(gejala_terpilih)
            ]

    return render_template(
        "index.html",
        symptoms=SYMPTOMS,
        symptom_label=SYMPTOM_LABEL,
        symptom_icon=SYMPTOM_ICON,
        gejala_terpilih=gejala_terpilih,
        hasil_fc=hasil_fc,
        hasil_bc=hasil_bc,
        hasil_nb=hasil_nb,
        submitted=submitted,
    )


if __name__ == "__main__":
    app.run(debug=True)