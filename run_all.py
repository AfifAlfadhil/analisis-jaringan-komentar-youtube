"""
run_all.py
==========
Menjalankan tahap 2–9 berurutan (pengambilan data / main.py TIDAK ikut, karena memakai kuota API).
Tahap 2 membutuhkan preprocess.py, yang tidak ada pada checkout ini; gunakan --from 5
untuk menjalankan community detection dan tahap setelahnya pada data siap pakai.

    python run_all.py --from 5        # mulai dari deteksi komunitas sampai akhir
    python run_all.py --from 6        # mulai dari join komentar-komunitas sampai akhir
    python run_all.py --only 8 9      # hanya tahap tertentu
    python run_all.py --fast          # tanpa BERTopic & IndoBERT (TF-IDF + leksikon saja)

Berhenti di tahap pertama yang gagal.
"""

import argparse
import subprocess
import sys
import time

STAGES = [
    (2, "Preprocessing", ["preprocess.py"]),
    (3, "Bangun graf", ["build_graph.py"]),
    (4, "Analisis jaringan (SNA)", ["analyze_network.py"]),
    (5, "Deteksi komunitas", ["-m", "stages.community_detection.detect_communities"]),
    (6, "Join komentar-komunitas", ["-m", "stages.community_detection.join_communities"]),
    (7, "Nama tokoh per komunitas", ["-m", "stages.topic_extraction.extract_entities"]),
    (7, "Pemetaan isu (TF-IDF + BERTopic)", ["-m", "stages.topic_extraction.topic_modeling"]),
    (8, "Analisis sentimen", ["-m", "stages.sentiment_analysis.sentiment_analysis"]),
    (9, "Perbandingan antar komunitas", ["-m", "stages.community_comparison.compare_communities"]),
]
FAST_FLAGS = {
    "stages.topic_extraction.topic_modeling": ["--no-bertopic"],
    "stages.sentiment_analysis.sentiment_analysis": ["--no-model"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", type=int, default=2)
    p.add_argument("--only", type=int, nargs="+")
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()

    todo = [s for s in STAGES if (s[0] in args.only if args.only else s[0] >= args.start)]
    t0 = time.time()
    for num, name, cmd in todo:
        script = cmd[-1]
        full = [sys.executable, *cmd, *(FAST_FLAGS.get(script, []) if args.fast else [])]
        print(f"\n{'=' * 70}\nTahap {num} — {name}: {' '.join(full[1:])}\n{'=' * 70}", flush=True)
        t = time.time()
        if subprocess.run(full).returncode != 0:
            sys.exit(f"\nGagal di Tahap {num} ({script}). Perbaiki error di atas lalu jalankan: "
                     f"python run_all.py --from {num}")
        print(f"[selesai dalam {time.time() - t:.0f} detik]")
    print(f"\nSemua tahap selesai dalam {time.time() - t0:.0f} detik.")


if __name__ == "__main__":
    main()
