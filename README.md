# Analisis Jaringan Interaksi Publik, Pemetaan Isu, dan Analisis Sentimen Pilpres 2029 (Komentar YouTube)

Final project Analisis Media Sosial. Proposal: `docs/rencana_projek.txt`. Rencana pengerjaan: `docs/plan_pengerjaan_final_project.md`.

## Menjalankan

```bash
pip install -r requirements.txt
pip install -r requirements-nlp.txt   # BERTopic + IndoBERT (Tahap 7–8), ukuran besar
# isi YOUTUBE_API_KEY di .env (salin dari .env.example)

python main.py                 # Tahap 1: kumpulkan video + komentar + reply -> data/raw/
python preprocess.py           # Tahap 2: -> data/processed/comments_clean.csv, author_map.csv (privat)
python build_graph.py          # Tahap 3: -> data/processed/edges.csv, outputs/graph/network.graphml
python analyze_network.py      # Tahap 4: -> network_stats, centrality, top_actors, actor_roles, figures/
python detect_communities.py   # Tahap 5: -> community_{sweep,assignment,stats,summary}.csv
python join_communities.py     # Tahap 6: -> data/processed/comments_with_community.csv, join_summary.csv
python extract_entities.py     # Tahap 7: nama tokoh per komunitas -> entity_*.csv, heatmap
python topic_modeling.py       # Tahap 7: TF-IDF (community_keywords) + BERTopic (topics, topic_by_community)
python sentiment_analysis.py --make-sample   # Tahap 8a: sampel 400 komentar untuk dilabeli manual
python sentiment_analysis.py   # Tahap 8b: leksikon + IndoBERT, evaluasi bila label manual sudah diisi
python compare_communities.py  # Tahap 9: sentimen per komunitas, chi-square, Cramér's V, uji permutasi
```

Dashboard interaktif (membaca hasil di `outputs/`, tidak menjalankan ulang model):
`streamlit run dashboard.py` lalu buka alamat yang muncul (biasanya http://localhost:8501).

Atau sekaligus: `python run_all.py` (Tahap 2–9), `python run_all.py --from 6`, atau
`python run_all.py --fast` (tanpa BERTopic/IndoBERT, untuk uji cepat).

## Tahap 6–9 — ringkasan

| Tahap | Skrip | Output utama | Catatan |
|---|---|---|---|
| 6 | `join_communities.py` | `comments_with_community.csv` | Komentar ikut komunitas penulisnya; penulis di luar graf = `tanpa_interaksi` (pembanding). |
| 7 | `extract_entities.py` | `entity_top_by_community.csv`, `entity_pairs_by_community.csv` | Kamus tokoh di `src/entities.py`; cek `candidate_discovery.csv` untuk alias baru. |
| 7 | `topic_modeling.py` | `community_keywords.csv` (c-TF-IDF), `topics.csv`, `themes.csv`, `theme_by_community.csv` | Beri nama topik di `topic_labels.csv` (kolom `label_manual`) lalu jalankan ulang; topik bernama sama digabung menjadi satu tema. |
| 8 | `sentiment_analysis.py` | `sentiment_predictions.csv`, `sentiment_eval.csv`, `sentiment_agreement.csv` | Model default di `config.SENTIMENT_MODEL`. Label manual di `data/processed/sentiment_label_sample.csv`. |
| 9 | `compare_communities.py` | `sentiment_by_community.csv`, `sentiment_tests.csv`, `entity_sentiment_by_community.csv` | Versi komentar & versi akun; uji permutasi di dalam video mengontrol efek video. |

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
├── preprocess.py             # Tahap 2
├── build_graph.py            # Tahap 3
├── analyze_network.py        # Tahap 4
├── detect_communities.py     # Tahap 5
├── join_communities.py       # Tahap 6
├── extract_entities.py       # Tahap 7 (nama tokoh)
├── topic_modeling.py         # Tahap 7 (TF-IDF + BERTopic)
├── sentiment_analysis.py     # Tahap 8
├── compare_communities.py    # Tahap 9
├── run_all.py                # menjalankan Tahap 2–9
├── dashboard.py              # dashboard Streamlit (hasil Tahap 4–9)
├── src/
│   ├── youtube_collector.py  # YouTube API
│   ├── preprocessing.py
│   ├── lexicons.py           # slang, stopword, pola spam
│   ├── network.py
│   ├── sna.py                # statistik jaringan, centrality
│   ├── community.py          # Louvain/Leiden, kestabilan, statistik komunitas
│   ├── entities.py           # kamus tokoh & partai
│   ├── topics.py             # frekuensi kata, c-TF-IDF, BERTopic, koherensi
│   ├── sentiment.py          # leksikon, IndoBERT, metrik evaluasi
│   └── plots.py              # heatmap & bar sentimen
├── data/
│   ├── raw/                  # videos.csv, comments.csv (jangan diubah)
│   └── processed/            # comments_clean, edges, author_map (privat)
├── outputs/                  # graph/, tables/, figures/
└── docs/                     # proposal dan plan
```

## Etika

Akun dianonimkan (`user_0001`, dst.) di semua data olahan. `data/processed/author_map.csv` memetakan ke channel ID dan handle asli, jangan dipublikasikan atau dimasukkan ke laporan/slide. Analisis berada di level kelompok, bukan individu.
