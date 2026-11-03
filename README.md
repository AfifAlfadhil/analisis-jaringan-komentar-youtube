# Analisis Jaringan Interaksi Publik, Pemetaan Isu, dan Analisis Sentimen Pilpres 2029 (Komentar YouTube)

Final project Analisis Media Sosial. Proposal: `docs/rencana_projek.txt`. Rencana pengerjaan: `docs/plan_pengerjaan_final_project.md`.

## Menjalankan

```bash
pip install -r requirements.txt
# isi YOUTUBE_API_KEY di .env (salin dari .env.example)

python main.py               # Tahap 1: kumpulkan video + komentar + reply -> data/raw/
python preprocess.py         # Tahap 2: -> data/processed/comments_clean.csv, author_map.csv (privat)
python build_graph.py        # Tahap 3: -> data/processed/edges.csv, outputs/graph/network.graphml
python analyze_network.py    # Tahap 4: -> outputs/tables/{network_stats,centrality,top_actors}.csv, outputs/figures/
python detect_communities.py # Tahap 5: -> outputs/tables/community_{sweep,assignment,stats,summary}.csv
```

## Tahap 1 — Pengumpulan data

`python main.py` memakai 5 kata kunci dari proposal (Pilpres 2029, Kandidat Presiden 2029, Calon Presiden 2029, Prediksi Pilpres 2029, Bursa Capres 2029) dan parameter default di `config.py`.

Seleksi video: dipublikasikan dalam N hari terakhir, komentar aktif, jumlah komentar >= batas minimum, bahasa Indonesia (bila metadata bahasa tersedia). Video yang ditemukan beberapa kata kunci digabung dan kolom `keyword` berisi semuanya.

Komentar: `commentThreads.list` untuk komentar tingkat atas, lalu `comments.list(parentId)` untuk mengambil **semua** reply thread (commentThreads sendiri hanya menyertakan maks 5 reply).

| Opsi | Default | Keterangan |
|---|---|---|
| `-k`, `--keyword` | 5 kata kunci proposal | boleh diulang |
| `--max-videos` | 15 | video lolos seleksi per kata kunci |
| `--days` | 120 | rentang publikasi (atau `--published-after/--published-before`) |
| `--min-comments` | 50 | komentar minimum per video |
| `--max-threads` | 300 | maks thread tingkat atas per video (semua reply tetap diambil) |
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
├── preprocess.py             # Tahap 2
├── build_graph.py            # Tahap 3
├── analyze_network.py        # Tahap 4
├── detect_communities.py     # Tahap 5
├── src/
│   ├── youtube_collector.py  # YouTube API
│   ├── preprocessing.py
│   ├── lexicons.py           # slang, stopword, pola spam
│   ├── network.py
│   ├── sna.py                # statistik jaringan, centrality
│   └── community.py          # Louvain/Leiden, kestabilan, statistik komunitas
├── data/
│   ├── raw/                  # videos.csv, comments.csv (jangan diubah)
│   └── processed/            # comments_clean, edges, author_map (privat)
├── outputs/                  # graph/, tables/, figures/
└── docs/                     # proposal dan plan
```

## Etika

Akun dianonimkan (`user_0001`, dst.) di semua data olahan. `data/processed/author_map.csv` memetakan ke channel ID dan handle asli, jangan dipublikasikan atau dimasukkan ke laporan/slide. Analisis berada di level kelompok, bukan individu.
