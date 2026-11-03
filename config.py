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
#   data/processed  : data bersih, edges, komentar + komunitas
#   outputs/        : tabel, graf, dan gambar hasil analisis
# --------------------------------------------------------------------------- #
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
GRAPH_DIR = BASE_DIR / "outputs" / "graph"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
for _d in (RAW_DIR, PROCESSED_DIR, TABLES_DIR, GRAPH_DIR, FIGURES_DIR):
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

# --------------------------------------------------------------------------- #
# Community detection (Tahap 5)
# --------------------------------------------------------------------------- #
COMMUNITY_ALGORITHM = "leiden"             # algoritma final: "leiden" atau "louvain"
COMMUNITY_RESOLUTION = 1.0                 # resolution final (lihat outputs/tables/community_sweep.csv)
COMMUNITY_RESOLUTIONS = [0.5, 0.75, 1.0, 1.5, 2.0]   # nilai yang dicoba pada sweep
COMMUNITY_N_SEEDS = 20                     # jumlah seed untuk uji kestabilan
COMMUNITY_MIN_SIZE = 20                    # komunitas lebih kecil digabung jadi "small"

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
