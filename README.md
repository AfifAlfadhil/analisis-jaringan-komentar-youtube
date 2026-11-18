# Analisis Komentar YouTube

Pipeline ini menganalisis jaringan reply, komunitas diskusi, isu, dan sentimen
komentar YouTube. Setiap tahap memiliki runner sendiri di `scripts/`.

## Tahap dan input-output

| Tahap | Runner | Input | Output |
|---|---|---|---|
| Pengumpulan | `scripts/collect_data.py` | YouTube Data API | `data/raw/` |
| Preprocessing | `scripts/preprocess.py` | raw comments | `data/processed/preprocessing/` |
| Graf dan SNA | `scripts/build_graph.py`, `scripts/analyze_network.py` | raw comments, videos | `data/processed/network_analysis/`, `outputs/network_analysis/` |
| Community detection | `scripts/detect_communities.py` | graph, raw comments, videos | `outputs/community_detection/` |
| Join | `scripts/join_communities.py` | cleaned comments, assignment komunitas | `data/processed/community_detection/` |
| Ekstraksi isu | `scripts/topic_modeling.py` | komentar hasil join | `outputs/topic_extraction/`, `data/processed/topic_extraction/` |
| Ekstraksi entitas | `scripts/extract_entities.py` | komentar hasil join | `outputs/topic_extraction/` |
| Sentimen | `scripts/sentiment_analysis.py` | komentar hasil join | `outputs/sentiment_analysis/`, `data/processed/sentiment_analysis/` |
| Perbandingan | `scripts/compare_communities.py` | hasil sentimen, isu, dan entitas | `outputs/community_comparison/` |

Community detection tidak memakai hasil preprocessing teks, isu, atau sentimen.
Tahap tersebut hanya memakai hubungan reply `penulis reply -> penulis komentar
induk`.

## Menjalankan

```bash
pip install -r requirements.txt
python scripts/collect_data.py
python scripts/preprocess.py
python scripts/build_graph.py
python scripts/analyze_network.py
python scripts/detect_communities.py
python scripts/join_communities.py
python scripts/topic_modeling.py
python scripts/extract_entities.py
python scripts/sentiment_analysis.py --make-sample
python scripts/sentiment_analysis.py
python scripts/compare_communities.py
```

BERTopic digunakan untuk ekstraksi isu/topik. Pasang `requirements-nlp.txt`
sebelum menjalankan `scripts/topic_modeling.py`. Opsi `--no-bertopic` hanya
menjalankan ringkasan kata kunci c-TF-IDF.

Dashboard Streamlit adalah tambahan visualisasi. Dashboard hanya membaca
output yang sudah dibuat, tidak menjalankan preprocessing, BERTopic, atau
sentimen:

```bash
streamlit run dashboard.py
```

## Struktur

```text
src/       fungsi reusable per tahap
scripts/   runner tahap
data/raw/  data sumber
data/processed/  data antara
outputs/   hasil analisis dan graf
docs/      proposal dan panduan
```
