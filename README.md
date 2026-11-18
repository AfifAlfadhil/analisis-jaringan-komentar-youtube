# Analisis Jaringan Interaksi Publik, Pemetaan Isu, dan Analisis Sentimen Pilpres 2029 (Komentar YouTube)

Final project Analisis Media Sosial. Proposal: `docs/rencana_projek.txt`. Rencana pengerjaan: `docs/plan_pengerjaan_final_project.md`.

## Menjalankan

```bash
pip install -r requirements.txt
pip install -r requirements-nlp.txt   # BERTopic + IndoBERT (Tahap 7–8), ukuran besar
# isi YOUTUBE_API_KEY di .env (salin dari .env.example)

python main.py                 # Tahap 1: kumpulkan video + komentar + reply -> data/raw/
# Tahap 2: siapkan data/processed/comments_clean.csv dan author_map.csv (privat)
python build_graph.py          # Tahap 3: -> data/processed/edges.csv, outputs/graph/network.graphml
python analyze_network.py      # Tahap 4: -> network_stats, centrality, top_actors, actor_roles, figures/
python -m stages.community_detection.detect_communities
python -m stages.community_detection.join_communities
python -m stages.topic_extraction.extract_entities
python -m stages.topic_extraction.topic_modeling
python -m stages.sentiment_analysis.sentiment_analysis --make-sample
python -m stages.sentiment_analysis.sentiment_analysis
python -m stages.community_comparison.compare_communities
```

Perintah lama seperti `python detect_communities.py` tetap didukung sebagai wrapper.
Dashboard interaktif (membaca hasil di `outputs/`, tidak menjalankan ulang model):
`streamlit run dashboard.py` lalu buka alamat yang muncul (biasanya http://localhost:8501).

Atau sekaligus: `python run_all.py --from 5` (community detection sampai Tahap 9),
`python run_all.py --from 6`, atau `python run_all.py --from 5 --fast`
(tanpa BERTopic/IndoBERT, untuk uji cepat). Checkout ini tidak menyertakan `preprocess.py`,
jadi tahap preprocessing harus disediakan sebelum menjalankan `run_all.py` dari Tahap 2.

## Tahap 6–9 — ringkasan

| Tahap | Skrip | Output utama | Catatan |
|---|---|---|---|
| 6 | `stages/community_detection/join_communities.py` | `data/processed/community_detection/comments_with_community.csv` | Membaca assignment dengan kolom `author_id`, `community_id`; akun tanpa assignment menjadi `tanpa_interaksi`. |
| 7 | `stages/topic_extraction/extract_entities.py` | `outputs/topic_extraction/tables/entity_*.csv` | Kamus tokoh di `stages/topic_extraction/entities.py`; cek `candidate_discovery.csv` untuk alias baru. |
| 7 | `stages/topic_extraction/topic_modeling.py` | `outputs/topic_extraction/tables/` | c-TF-IDF, BERTopic, tema; edit `topic_labels.csv` lalu jalankan ulang. |
| 8 | `stages/sentiment_analysis/sentiment_analysis.py` | `outputs/sentiment_analysis/tables/` | Prediksi dan evaluasi; sampel label ada di `data/processed/sentiment_analysis/`. |
| 9 | `stages/community_comparison/compare_communities.py` | `outputs/community_comparison/` | Mengambil assignment terbaru dari hasil Tahap 6 berdasarkan `comment_id`, lalu membandingkan sentimen, topik, dan tokoh. |

Pedoman label manual sentimen: **positif** = mendukung/memuji/berharap baik; **negatif** = menolak,
mengkritik, menghina, menyindir (sarkasme dilabeli sesuai maksudnya); **netral** = pertanyaan,
informasi, prediksi tanpa sikap. Isi `label_1` (dan `label_2` bila ada pelabel kedua untuk Cohen's kappa).

## Tahap 1 — Pengumpulan data

`python main.py` memakai 5 kata kunci dari proposal (Pilpres 2029, Kandidat Presiden 2029, Calon Presiden 2029, Prediksi Pilpres 2029, Bursa Capres 2029) dan parameter default di `config.py`.

Seleksi video: dipublikasikan dalam N hari terakhir, komentar aktif, jumlah komentar >= batas minimum, bahasa Indonesia (bila metadata bahasa tersedia). Video yang ditemukan beberapa kata kunci digabung dan kolom `keyword` berisi semuanya.

Komentar: `commentThreads.list` untuk komentar tingkat atas, lalu `comments.list(parentId)` untuk mengambil **semua** reply thread (commentThreads sendiri hanya menyertakan maks 5 reply).

| Opsi | Default | Keterangan |
|---|---|---|
| `-k`, `--keyword` | 5 kata kunci proposal | boleh diulang |
| `--max-videos` | 20 | video lolos seleksi per kata kunci |
| `--days` | 300 | rentang publikasi (atau `--published-after/--published-before`) |
| `--min-comments` | 50 | komentar minimum per video |
| `--max-threads` | 100 | maks thread tingkat atas per video (semua reply tetap diambil) |
| `--order` | `relevance` | `relevance` atau `time` |
| `--skip-search` | - | lanjutkan ambil komentar dari `data/raw/videos.csv` tanpa mencari ulang |

Pengambilan komentar inkremental: video yang sudah ada di `comments.csv` dilewati, jadi aman dijalankan ulang setelah kuota harian (10.000 unit; `search.list` = 100 unit/halaman) habis. Mencari ulang akan menambah video baru ke `videos.csv` tanpa menghapus yang lama. Catat tanggal pengambilan: kolom `collected_at`.

Skema `data/raw/videos.csv`: `video_id, title, channel, channel_id, published_at, views, likes, comment_count, description, keyword, language, duration, collected_at`.

Skema `data/raw/comments.csv` (satu baris = satu komentar atau reply): `comment_id, video_id, parent_id (kosong bila tingkat atas), author_id (channel ID), author_name, text, published_at, like_count, reply_count, collected_at`.

## Struktur direktori

```text
.
├── .env.example / .env       # API key (.env jangan dibagikan)
├── config.py                 # path dan parameter terpusat
├── main.py                   # Tahap 1: pengambilan data (CLI)
├── build_graph.py            # Tahap 3
├── analyze_network.py        # Tahap 4
├── stages/
│   ├── community_detection/   # deteksi komunitas + join komentar
│   ├── topic_extraction/      # skrip, kamus entitas, dan logika topik
│   ├── sentiment_analysis/    # skrip dan logika prediksi/evaluasi
│   └── community_comparison/  # analisis lintas komunitas
├── detect_communities.py      # wrapper kompatibilitas
├── join_communities.py        # wrapper kompatibilitas
├── extract_entities.py        # wrapper kompatibilitas
├── topic_modeling.py          # wrapper kompatibilitas
├── sentiment_analysis.py      # wrapper kompatibilitas
├── compare_communities.py     # wrapper kompatibilitas
├── run_all.py                # menjalankan Tahap 2–9
├── dashboard.py              # dashboard Streamlit (hasil Tahap 4–9)
├── src/
│   ├── youtube_collector.py  # YouTube API
│   ├── network.py
│   ├── sna.py                # statistik jaringan, centrality
│   ├── community.py          # Louvain/Leiden, kestabilan, statistik komunitas
│   └── plots.py              # heatmap & bar sentimen
├── data/
│   ├── raw/                  # videos.csv, comments.csv (jangan diubah)
│   └── processed/
│       ├── community_detection/
│       ├── topic_extraction/
│       └── sentiment_analysis/
├── outputs/
│   ├── community_detection/
│   ├── topic_extraction/
│   ├── sentiment_analysis/
│   ├── community_comparison/
│   ├── graph/                # graf jaringan
│   ├── figures/              # visualisasi jaringan (Tahap 4)
│   └── tables/               # tabel analisis jaringan (Tahap 4)
└── docs/                     # proposal dan plan
```

## Etika

Akun dianonimkan (`user_0001`, dst.) di semua data olahan. `data/processed/author_map.csv` memetakan ke channel ID dan handle asli, jangan dipublikasikan atau dimasukkan ke laporan/slide. Analisis berada di level kelompok, bukan individu.
