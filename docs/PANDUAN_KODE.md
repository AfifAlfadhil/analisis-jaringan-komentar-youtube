# Panduan Kode

Runner resmi berada di `scripts/`. Wrapper dengan nama sama di root tidak
dipertahankan agar tidak ada dua cara menjalankan tahap yang sama.

## Urutan tahap

1. `python scripts/collect_data.py`
2. `python scripts/preprocess.py`
3. `python scripts/build_graph.py`
4. `python scripts/analyze_network.py`
5. `python scripts/detect_communities.py`
6. `python scripts/join_communities.py`
7. `python scripts/topic_modeling.py`
8. `python scripts/extract_entities.py`
9. `python scripts/sentiment_analysis.py`
10. `python scripts/compare_communities.py`

Tahap 5 dapat dijalankan mandiri setelah graf tersedia. Tahap 7, 8, dan 9
dapat dijalankan terpisah setelah tahap join selesai.

## Ekstraksi isu

`topic_modeling.py` memakai:

- **BERTopic** untuk mengelompokkan komentar menjadi topik.
- **c-TF-IDF** untuk kata kunci khas setiap komunitas.
- `topics.csv`, `comment_topics.csv`, dan `topic_by_community.csv` sebagai
  hasil topik.

Jika hanya ingin memeriksa kata kunci tanpa menjalankan model:

```bash
python scripts/topic_modeling.py --no-bertopic
```

## Sentimen

`sentiment_analysis.py` mengklasifikasikan komentar ke tiga kelas:
`positif`, `netral`, dan `negatif` menggunakan leksikon. Sampel label manual
dapat dibuat dengan `--make-sample`, lalu dipakai untuk menghitung evaluasi.

## Dashboard

`dashboard.py` hanya membaca CSV dan GraphML dari `data/processed/` serta
`outputs/`. Tidak ada pemrosesan data atau pemanggilan model di dalamnya.

## Data privat

`data/processed/network_analysis/author_map.csv` memetakan ID anonim ke channel asli. File ini
tidak boleh dipublikasikan.
