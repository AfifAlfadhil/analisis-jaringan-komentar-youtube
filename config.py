"""
config.py
=========
Konfigurasi terpusat untuk final project:
Analisis Jaringan Interaksi Publik, Pemetaan Isu, dan Analisis Sentimen
Terkait Spekulasi Pilpres 2029 pada Diskusi Komentar YouTube.

Parameter pengambilan data mengacu pada docs/rencana_projek.txt dan
docs/plan_pengerjaan_final_project.md (Tahap 1).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# UTF-8 untuk terminal Windows agar emoji & teks Unicode tidak error
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --------------------------------------------------------------------------- #
# YouTube Data API v3
# --------------------------------------------------------------------------- #
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# --------------------------------------------------------------------------- #
# Direktori
#   data/raw        : hasil API (jangan diubah manual)
#   data/processed  : data bersih, edges, dan data perantara tiap tahap
#   outputs/        : hasil analisis yang dipisah menurut tahap
# --------------------------------------------------------------------------- #
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
COMMUNITY_PROCESSED_DIR = PROCESSED_DIR / "community_detection"
TOPIC_PROCESSED_DIR = PROCESSED_DIR / "topic_extraction"
SENTIMENT_PROCESSED_DIR = PROCESSED_DIR / "sentiment_analysis"
OUTPUTS_DIR = BASE_DIR / "outputs"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
GRAPH_DIR = BASE_DIR / "outputs" / "graph"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
COMMUNITY_TABLES_DIR = OUTPUTS_DIR / "community_detection" / "tables"
TOPIC_TABLES_DIR = OUTPUTS_DIR / "topic_extraction" / "tables"
SENTIMENT_TABLES_DIR = OUTPUTS_DIR / "sentiment_analysis" / "tables"
COMPARISON_TABLES_DIR = OUTPUTS_DIR / "community_comparison" / "tables"
TOPIC_FIGURES_DIR = OUTPUTS_DIR / "topic_extraction" / "figures"
COMPARISON_FIGURES_DIR = OUTPUTS_DIR / "community_comparison" / "figures"
for _d in (
    RAW_DIR,
    PROCESSED_DIR,
    COMMUNITY_PROCESSED_DIR,
    TOPIC_PROCESSED_DIR,
    SENTIMENT_PROCESSED_DIR,
    TABLES_DIR,
    GRAPH_DIR,
    FIGURES_DIR,
    COMMUNITY_TABLES_DIR,
    TOPIC_TABLES_DIR,
    SENTIMENT_TABLES_DIR,
    COMPARISON_TABLES_DIR,
    TOPIC_FIGURES_DIR,
    COMPARISON_FIGURES_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

RAW_VIDEOS_PATH = RAW_DIR / "videos.csv"
RAW_COMMENTS_PATH = RAW_DIR / "comments.csv"

# --------------------------------------------------------------------------- #
# Pengambilan data (Tahap 1)
# --------------------------------------------------------------------------- #
SEARCH_KEYWORDS = [
    "Pilpres 2029",
    "Kandidat Presiden 2029",
    "Calon Presiden 2029",
    "Prediksi Pilpres 2029",
    "Bursa Capres 2029",
]

DEFAULT_REGION_CODE = "ID"
DEFAULT_RELEVANCE_LANGUAGE = "id"

# Kriteria seleksi video
DEFAULT_MAX_VIDEOS_PER_KEYWORD = 20   # video lolos seleksi per kata kunci
DEFAULT_DAYS = 300                    # rentang publikasi: N hari terakhir
DEFAULT_MIN_COMMENTS = 50             # jumlah komentar minimum per video
MAX_SEARCH_PAGES = 3                  # batas halaman search per kata kunci (100 unit/halaman)

# Komentar
DEFAULT_MAX_THREADS_PER_VIDEO = 100   # batas thread (komentar tingkat atas) per video; semua reply tiap thread diambil
DEFAULT_COMMENT_ORDER = "relevance"   # "relevance" atau "time"

# Community detection
COMMUNITY_ALGORITHM = "leiden"
COMMUNITY_RESOLUTION = 1.0
COMMUNITY_RESOLUTIONS = [0.5, 0.75, 1.0, 1.5, 2.0]
COMMUNITY_N_SEEDS = 20
COMMUNITY_MIN_SIZE = 20

# --------------------------------------------------------------------------- #
# Tahap 6 — Join komentar ke komunitas
# --------------------------------------------------------------------------- #
LABEL_SMALL = "kecil"                       # label dari src.community untuk komunitas kecil
LEGACY_LABEL_SMALL = "small"
LABEL_NO_INTERACTION = "tanpa_interaksi"    # akun yang tidak masuk graf (tidak pernah membalas/dibalas)


def normalize_community_label(label):
    """Map the earlier small-community label to the current community output label."""
    if isinstance(label, str) and label == LEGACY_LABEL_SMALL:
        return LABEL_SMALL
    return label

# --------------------------------------------------------------------------- #
# Tahap 7 — Pemetaan isu (TF-IDF + BERTopic)
# --------------------------------------------------------------------------- #
TOPIC_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"   # sentence-transformers
TOPIC_MIN_TOPIC_SIZE = 30        # ukuran minimum topik BERTopic (HDBSCAN min_cluster_size)
TOPIC_NR_TOPICS = 20             # gabungkan topik mirip sampai maks. 20; None = biarkan BERTopic menentukan; "auto" = otomatis
TOPIC_TOP_WORDS = 10             # kata kunci per topik / per komunitas
TOPIC_SEED = 42

# --------------------------------------------------------------------------- #
# Tahap 8 — Analisis sentimen
# --------------------------------------------------------------------------- #
# Model IndoBERT hasil fine-tune SmSA (IndoNLU), 3 kelas. Cek kartu model di Hugging Face.
# SENTIMENT_MODEL = "hanifnoerr/Fine-tuned-Indonesian-Sentiment-Classifier"
SENTIMENT_MODEL = "taufiqdp/indonesian-sentiment"
SENTIMENT_LABEL_MAP = {"LABEL_0": "negatif", "LABEL_1": "netral", "LABEL_2": "positif"}
# SENTIMENT_MODEL = "mdhugol/indonesia-bert-sentiment-classification"
# Pemetaan label keluaran model -> positif/netral/negatif. Label bernama (positive, negatif, ...)
# dikenali otomatis; isi di sini hanya untuk label generik seperti LABEL_0.
# SENTIMENT_LABEL_MAP = {"LABEL_0": "positif", "LABEL_1": "netral", "LABEL_2": "negatif"}
SENTIMENT_MAX_LENGTH = 128
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_SAMPLE_SIZE = 400      # jumlah komentar untuk dilabeli manual (validasi)
SENTIMENT_SEED = 42
SENTIMENT_LABELS = ["positif", "netral", "negatif"]

# --------------------------------------------------------------------------- #
# Tahap 9 — Perbandingan antar komunitas
# --------------------------------------------------------------------------- #
COMPARE_N_PERMUTATIONS = 2000    # uji permutasi label komunitas di dalam video
COMPARE_SEED = 42

# Biaya kuota YouTube Data API (unit)
QUOTA_COST_SEARCH = 100
QUOTA_COST_LIST = 1


def validate_api_key() -> str:
    """Validasi keberadaan API key YouTube Data API v3."""
    if not YOUTUBE_API_KEY or "api_key" in YOUTUBE_API_KEY.lower():
        raise EnvironmentError(
            "YOUTUBE_API_KEY belum diisi atau tidak valid.\n"
            "Salin .env.example menjadi .env lalu isi:\n"
            "    YOUTUBE_API_KEY=AIzaSyxxxxxxx\n"
            "API key: https://console.cloud.google.com/apis/credentials"
        )
    return YOUTUBE_API_KEY
