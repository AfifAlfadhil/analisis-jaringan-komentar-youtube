# Analisis Jaringan Komentar YouTube

Pipeline ini hanya membangun jaringan reply dan mendeteksi komunitas akun.

## Menjalankan

```bash
pip install -r requirements.txt
python main.py
python build_graph.py
python analyze_centrality.py
python detect_communities.py
```

Untuk melanjutkan pengumpulan komentar setelah kuota API habis:

```bash
python main.py --skip-search
python build_graph.py
python detect_communities.py
```

## Data

Data mentah dipakai langsung:

- `data/raw/comments.csv`: komentar tingkat atas dan reply
- `data/raw/videos.csv`: metadata video

Kolom yang digunakan untuk membangun jaringan:

- `comment_id`: identitas komentar
- `author_id`: channel ID penulis
- `author_name`: nama tampilan, dipakai hanya sebagai fallback
- `parent_id`: komentar induk yang dibalas
- `video_id`: identitas video

## Luaran

- `data/processed/edges.csv`: edge `source -> target` beserta bobot
- `data/processed/edges_selfloop.csv`: balasan akun kepada dirinya sendiri
- `data/processed/author_map.csv`: pemetaan channel ID ke ID anonim
- `outputs/graph/network.graphml`
- `outputs/graph/network.gexf`
- `outputs/tables/community_assignment.csv`: pemetaan akun ke komunitas
- `outputs/tables/community_stats.csv`: ukuran dan hub tiap komunitas
- `outputs/tables/centrality.csv`: metrik sentralitas akun

## Definisi jaringan

Setiap reply membentuk satu edge:

```text
penulis reply -> penulis komentar induk
```

Bobot edge adalah jumlah reply dari pasangan akun yang sama. Self-loop tidak dimasukkan ke graf komunitas.
