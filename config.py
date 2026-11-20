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
PREPROCESSING_PROCESSED_DIR = PROCESSED_DIR / "preprocessing"
NETWORK_PROCESSED_DIR = PROCESSED_DIR / "network_analysis"
COMMUNITY_PROCESSED_DIR = PROCESSED_DIR / "community_detection"
TOPIC_PROCESSED_DIR = PROCESSED_DIR / "topic_extraction"
SENTIMENT_PROCESSED_DIR = PROCESSED_DIR / "sentiment_analysis"
SENTIMENT_LLM_CACHE_DIR = SENTIMENT_PROCESSED_DIR / "llm_cache"
OUTPUTS_DIR = BASE_DIR / "outputs"
NETWORK_OUTPUT_DIR = OUTPUTS_DIR / "network_analysis"
NETWORK_GRAPH_DIR = NETWORK_OUTPUT_DIR / "graph"
NETWORK_FIGURES_DIR = NETWORK_OUTPUT_DIR / "figures"
COMMUNITY_OUTPUT_DIR = OUTPUTS_DIR / "community_detection"
TOPIC_OUTPUT_DIR = OUTPUTS_DIR / "topic_extraction"
SENTIMENT_OUTPUT_DIR = OUTPUTS_DIR / "sentiment_analysis"
COMPARISON_OUTPUT_DIR = OUTPUTS_DIR / "community_comparison"
TOPIC_FIGURES_DIR = OUTPUTS_DIR / "topic_extraction" / "figures"
COMPARISON_FIGURES_DIR = OUTPUTS_DIR / "community_comparison" / "figures"
for _d in (
    RAW_DIR,
    PROCESSED_DIR,
    PREPROCESSING_PROCESSED_DIR,
    NETWORK_PROCESSED_DIR,
    COMMUNITY_PROCESSED_DIR,
    TOPIC_PROCESSED_DIR,
    SENTIMENT_PROCESSED_DIR,
    SENTIMENT_LLM_CACHE_DIR,
    NETWORK_GRAPH_DIR,
    NETWORK_FIGURES_DIR,
    COMMUNITY_OUTPUT_DIR,
    TOPIC_OUTPUT_DIR,
    SENTIMENT_OUTPUT_DIR,
    COMPARISON_OUTPUT_DIR,
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
LABEL_SMALL = "kecil"                       # label dari src.community_detection untuk komunitas kecil
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
# Tahap 8 — Analisis sentimen (3 LLM Anotator + Fleiss' Kappa + Majority Vote)
# --------------------------------------------------------------------------- #
SENTIMENT_SAMPLE_SIZE = 150      # jumlah komentar untuk pilot test Fleiss' Kappa
SENTIMENT_SEED = 42
SENTIMENT_LABELS = ["positif", "netral", "negatif"]
SENTIMENT_FLEISS_TARGET = 0.85   # syarat minimal Fleiss' Kappa (Landis & Koch: Almost Perfect >= 0.81)
SENTIMENT_TIE_BREAKER = "netral" # label fallback bila 3 anotator memilih 1-1-1 berbeda semua

# Konfigurasi 3 LLM Anotator (API key & endpoint dibaca dari .env)
# Pengaturan batch_size & request_delay disesuaikan per model sedikit di bawah batas aman API
LLM_ANNOTATORS = [
    {
        "id": 1,
        "name": os.getenv("ANNOTATOR_1_NAME", "gemini_3_5_flash").strip(),
        "provider": os.getenv("ANNOTATOR_1_PROVIDER", "gemini").strip(),
        "model": os.getenv("ANNOTATOR_1_MODEL", "gemini-3.5-flash").strip(),
        "base_url": os.getenv("ANNOTATOR_1_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").strip(),
        "api_key": (os.getenv("ANNOTATOR_1_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip(),
        "batch_size": 70,        # 6.601 / 70 = ~95 request (hemat RPD harian Gemini)
        "request_delay": 4.0,    # jeda antar batch (detik)
        "max_text_chars": 500,
        "tpm_budget": 0,
    },
    {
        "id": 2,
        "name": os.getenv("ANNOTATOR_2_NAME", "llama_3_3_70b").strip(),
        "provider": os.getenv("ANNOTATOR_2_PROVIDER", "openai_compatible").strip(),
        "model": os.getenv("ANNOTATOR_2_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast").strip(),
        "base_url": os.getenv("ANNOTATOR_2_BASE_URL", "").strip(),
        "api_key": os.getenv("ANNOTATOR_2_API_KEY", "").strip(),
        "batch_size": 55,        # 6.601 / 55 = ~121 request (aman untuk JSON output Cloudflare)
        "request_delay": 3.5,    # jeda antar batch (detik)
        "max_text_chars": 450,
        "tpm_budget": 0,
    },
    {
        "id": 3,
        "name": os.getenv("ANNOTATOR_3_NAME", "qwen_3_8_27b").strip(),
        "provider": os.getenv("ANNOTATOR_3_PROVIDER", "openai_compatible").strip(),
        "model": os.getenv("ANNOTATOR_3_MODEL", "qwen/qwen3.8-27b").strip(),
        "base_url": os.getenv("ANNOTATOR_3_BASE_URL", "https://api.groq.com/openai/v1").strip(),
        "api_key": (os.getenv("ANNOTATOR_3_API_KEY") or os.getenv("GROQ_API_KEY") or "").strip(),
        "batch_size": 20,        # aman di bawah limit TPM free tier Groq
        "request_delay": 6.0,    # jeda minimal antar batch (detik) + dynamic TPM pacing
        "max_text_chars": 400,
        "tpm_budget": 10000,
    },
]

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
