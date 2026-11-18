# Panduan Kode Final Project: Analisis Jaringan Komentar YouTube Pilpres 2029

Diperbarui 15 November 2026. Versi online (bisa diedit dan dikomentari): https://claude.ai/artifact/NiBEAiTMq7LwiUQFQz8q52

## 1. Ringkasan alur dan cara menjalankan

Proyek ini sekarang punya pipeline lengkap Tahap 1–9: dari pengambilan komentar YouTube sampai perbandingan isu dan sentimen antar komunitas. Tahap 1–5 sudah ada sebelumnya; Tahap 6–9, ekstraksi nama tokoh, dan beberapa perbaikan ditambahkan pada 15 November 2026. Semua perintah dijalankan dari folder `C:\analisis-jaringan-komentar-youtube` dengan venv aktif (`venv\Scripts\activate`).

| Tahap | Perintah | Hasil utama |
| --- | --- | --- |
| 1. Ambil data | `python main.py` | `data/raw/videos.csv`, `comments.csv` (pakai kuota API, jalankan sekali saja) |
| 2. Preprocessing | `python preprocess.py` | `data/processed/comments_clean.csv` |
| 3. Bangun graf | `python build_graph.py` | `outputs/graph/network.graphml`, `edges.csv` |
| 4. Analisis jaringan | `python analyze_network.py` | `centrality.csv`, `top_actors.csv`, `actor_roles.csv` |
| 5. Deteksi komunitas | `python detect_communities.py` | `community_assignment.csv`, `community_stats.csv` |
| 6. Join komentar–komunitas | `python join_communities.py` | `data/processed/comments_with_community.csv` |
| 7a. Nama tokoh | `python extract_entities.py` | `entity_top_by_community.csv`, heatmap tokoh |
| 7b. Topik/isu | `python topic_modeling.py` | `community_keywords.csv`, `themes.csv`, `theme_by_community.csv` |
| 8. Sentimen | `python sentiment_analysis.py` | `sentiment_predictions.csv`, `sentiment_eval.csv` |
| 9. Perbandingan | `python compare_communities.py` | `sentiment_by_community.csv`, `sentiment_tests.csv` |

Jalan pintas: `python run_all.py` menjalankan Tahap 2–9 berurutan. `python run_all.py --from 6` mulai dari Tahap 6. `python run_all.py --fast` melewati BERTopic dan IndoBERT (cepat, untuk uji).

Pustaka: `pip install -r requirements.txt` untuk dasar, lalu `pip install -r requirements-nlp.txt` untuk BERTopic dan IndoBERT (ukuran besar, ada torch). Semua tabel ada di `outputs/tables/`, gambar di `outputs/figures/`.

## 2. File baru yang ditambahkan

Ada 11 file kode baru: 6 skrip yang dijalankan langsung dan 5 modul pendukung di folder `src/`. Skrip di folder utama hanya mengatur alur (baca data, panggil fungsi, simpan hasil); logika hitungnya ada di `src/`.

| File | Jenis | Fungsi |
| --- | --- | --- |
| `join_communities.py` | Skrip Tahap 6 | Menempelkan label komunitas penulis ke setiap komentar |
| `extract_entities.py` | Skrip Tahap 7a | Mencari nama tokoh/partai di komentar, menghitungnya per komunitas |
| `topic_modeling.py` | Skrip Tahap 7b | TF-IDF kata khas per komunitas + BERTopic + penamaan topik |
| `sentiment_analysis.py` | Skrip Tahap 8 | Membuat sampel label, prediksi sentimen (leksikon + IndoBERT), evaluasi |
| `compare_communities.py` | Skrip Tahap 9 | Proporsi sentimen per komunitas, chi-square, Cramér's V, uji permutasi |
| `run_all.py` | Skrip bantu | Menjalankan Tahap 2–9 sekaligus, berhenti kalau ada yang gagal |
| `src/entities.py` | Modul | Kamus tokoh dan partai beserta alias/julukannya |
| `src/topics.py` | Modul | Fungsi frekuensi kata, c-TF-IDF, BERTopic, koherensi c_v |
| `src/sentiment.py` | Modul | Kamus leksikon, pemanggil IndoBERT, metrik evaluasi, Cohen's kappa |
| `src/plots.py` | Modul | Heatmap komunitas × kategori dan grafik batang sentimen |
| `requirements-nlp.txt` | Daftar pustaka | bertopic, sentence-transformers, transformers, torch, gensim |

File data yang dibuat skrip-skrip ini dan perlu diketahui:

- `data/processed/comments_with_community.csv`: komentar + kolom `community_id` (C01–C19, `small`, atau `tanpa_interaksi`).
- `data/processed/entity_mentions.csv`: satu baris per komentar × tokoh yang disebut.
- `data/processed/comment_topics.csv`: topik dan nama tema tiap komentar.
- `data/processed/sentiment_label_sample.csv`: 400 komentar berlabel untuk validasi model sentimen.
- `data/processed/sentiment_predictions.csv`: label dan probabilitas sentimen tiap komentar; kolom `sentiment` = label final.
- `data/processed/embeddings_*.npy`: cache embedding BERTopic (boleh dihapus, akan dibuat ulang).

## 3. File lama yang diubah

Enam file lama diubah; perubahan ini menambah fitur tanpa mengubah hasil Tahap 1–5 (statistik jaringan dan komunitas tetap sama persis).

| File | Apa yang diubah | Alasan |
| --- | --- | --- |
| `config.py` | Ditambah parameter Tahap 6–9: model embedding, `TOPIC_MIN_TOPIC_SIZE = 30`, `TOPIC_NR_TOPICS = 20`, model sentimen `mdhugol/indonesia-bert-sentiment-classification`, ukuran sampel label 400, jumlah permutasi 2000 | Semua parameter di satu tempat, mudah diganti |
| `src/sna.py` | Kolom degree ternormalisasi deg(v)/(n−1); kategori aktor influencer / broker / initiator; fungsi `actor_roles()`; peringkat tanpa pemilik channel | Menyesuaikan rumus dan istilah di BAB II |
| `analyze_network.py` | Ikut menyimpan `actor_roles.csv` dan `top_actors_tanpa_pemilik_channel.csv` | Pemilik channel mendominasi peringkat di videonya sendiri |
| `README.md` | Daftar perintah Tahap 6–9, tabel ringkasan, struktur folder baru | Dokumentasi |
| `.gitignore` | Mengabaikan `data/processed/embeddings_*.npy` | File cache besar, tidak perlu masuk git |
| `extract_entities.py`, `src/entities.py` | Alias baru hasil temuan data: fafa, girban, serli, sherly juanda, subiyanto, annis, Cak Nun, dr. Tifa | Supaya sebutan tokoh tidak terlewat |

Perbaikan karena error saat dijalankan di laptop:

- `src/topics.py`: koherensi c_v (gensim) dipaksa 1 proses (`processes=1`).
- `topic_modeling.py`: jumlah thread OpenBLAS dibatasi 2; opsi baru `--no-coherence` dan `--nr-topics`; topik bernama sama otomatis digabung menjadi satu tema; peringatan Pandas diperbaiki.
- `compare_communities.py`: tabel topik × sentimen memakai nama tema, bukan nomor topik.

## 4. Cara kerja tiap tahap baru

Prinsip utama: label isu dan sentimen menempel pada **komentar**, sedangkan akun hanya meneruskan label komunitasnya ke komentar yang ia tulis.

### Tahap 4 (tambahan): peran aktor

- **Degree ternormalisasi**: `degree_centrality = degree / (n − 1)`, sesuai rumus BAB II. Peringkatnya sama dengan degree mentah.
- **Kategori aktor**: influencer = in-degree, in-strength, PageRank, eigenvector; broker = betweenness; initiator = out-degree, out-strength.
- **actor_roles.csv**: akun yang masuk 1% teratas di salah satu kelompok metrik, dengan kolom `roles` (satu akun bisa punya dua peran, mis. "influencer, broker").

### Tahap 6: join komentar ke komunitas

1. Baca `comments_clean.csv` dan `community_assignment.csv`.
2. Gabungkan lewat `author_id`. Akun yang tidak ada di graf (tidak pernah membalas atau dibalas) diberi label `tanpa_interaksi` sebagai kelompok pembanding.
3. Hasil saat ini: 28% komentar dari 19 komunitas utama, 22% dari `small`, 50% dari `tanpa_interaksi`.

### Tahap 7a: nama tokoh per komunitas

1. Kamus di `src/entities.py` berisi nama baku + alias (nama panggilan, singkatan, salah ketik, julukan). Contoh: KDM dicari lewat "kdm", "dedi mulyadi", "kang dedi".
2. Setiap komentar (`text_light`, huruf kecil) dicocokkan dengan regex. Satu komentar dihitung satu kali per tokoh.
3. Per komunitas dihitung: jumlah komentar, `share_comments` (% komentar komunitas yang menyebut tokoh), `share_authors` (% akun), dan `lift` = porsi di komunitas ÷ porsi di seluruh data. Lift > 1,5 dengan minimal 5 komentar = tokoh khas komunitas itu.
4. `candidate_discovery.csv` mendaftar kata berhuruf kapital di tengah kalimat yang belum ada di kamus, untuk ditambahkan manual.

### Tahap 7b: topik/isu

1. **Eksplorasi**: unigram dan bigram terbanyak (`term_frequency.csv`).
2. **c-TF-IDF per komunitas**: semua komentar satu komunitas digabung jadi satu dokumen, lalu bobot kata = frekuensi relatif di komunitas × log(1 + rata-rata kata per komunitas ÷ frekuensi kata di semua komunitas). Hasil: 10 kata paling khas tiap komunitas (`community_keywords_wide.csv`).
3. **BERTopic** pada level komentar: kalimat (`text_light`) diubah jadi embedding dengan `paraphrase-multilingual-MiniLM-L12-v2` → UMAP mengecilkan dimensi → HDBSCAN mengelompokkan (min. 30 komentar per topik) → topik mirip digabung sampai maks. 20 → kata kunci topik dari `text_full` dengan c-TF-IDF. Komentar yang tidak masuk topik mana pun = topik `-1` (outlier).
4. **Koherensi c_v** (gensim) mengukur seberapa nyambung kata kunci tiap topik; hasil saat ini rata-rata 0,39.
5. **Penamaan**: nama manual di `topic_labels.csv`. Topik dengan nama sama digabung menjadi satu tema (`themes.csv`, `theme_by_community.csv`, heatmap).
6. Hasil saat ini: 19 topik, outlier 27%, digabung menjadi 16 tema + "Lainnya".

### Tahap 8: sentimen

1. **Baseline leksikon**: kamus kecil kata positif/negatif politik + penanganan negasi ("tidak bagus" → negatif). Skor > 0 positif, < 0 negatif, 0 netral. Kalau file InSet disimpan di `data/lexicon/positive.tsv` dan `negative.tsv`, kamus itu yang dipakai.
2. **IndoBERT** (`mdhugol/indonesia-bert-sentiment-classification`, fine-tune SmSA): setiap komentar diberi probabilitas positif/netral/negatif; label = probabilitas tertinggi.
3. **Evaluasi** terhadap 400 label di `sentiment_label_sample.csv`: accuracy, precision, recall, F1 macro, confusion matrix. Cohen's kappa dihitung kalau kolom `label_2` diisi.
4. Kolom `sentiment` di `sentiment_predictions.csv` = hasil IndoBERT (atau leksikon kalau IndoBERT tidak dijalankan). Kolom `sentiment_source` mencatat sumbernya.
5. Hasil saat ini (leksikon saja): accuracy 0,41, F1 macro 0,40. IndoBERT belum dijalankan.

### Tahap 9: perbandingan antar komunitas

1. **Proporsi sentimen** per komunitas, dua versi: per komentar, dan per akun (dirata-rata per akun dulu agar akun yang sangat aktif tidak mendominasi).
2. **Chi-square + Cramér's V** tiga kali: (a) antar 19 komunitas utama, (b) komunitas utama + small + tanpa_interaksi, (c) tiga kelompok besar. Pedoman V: < 0,1 sangat lemah, 0,1–0,3 lemah, 0,3–0,5 sedang, > 0,5 kuat.
3. **Adjusted residual** per sel: nilai > +2 atau < −2 = komunitas itu menyimpang dari rata-rata pada label sentimen tersebut.
4. **Uji permutasi di dalam video**: label komunitas diacak di dalam tiap video 2000 kali. Kalau p kecil, perbedaan sentimen antar komunitas tidak cukup dijelaskan oleh perbedaan video saja.
5. **Sentimen per tokoh dan per tema**: `entity_sentiment_by_community.csv` dan `topic_sentiment.csv`.
6. Grafik: `outputs/figures/sentiment_by_community.png`.

## 5. Langkah manual

Dua bagian butuh campur tangan manusia: penamaan topik dan label validasi sentimen. Keduanya sudah terisi, tapi bisa diperiksa dan diubah.

### Penamaan topik

1. Jalankan `python topic_modeling.py`. Muncul `topic_examples.csv` (8 contoh komentar per topik) dan `topic_labels.csv`.
2. Isi kolom `label_manual` di `topic_labels.csv` dengan nama yang menggambarkan isi topik. Topik yang mirip diberi nama sama supaya digabung; topik tak bermakna diberi nama "Lainnya". Simpan sebagai **CSV UTF-8**.
3. Jalankan lagi `python topic_modeling.py`. Topiknya tetap sama selama data, seed, dan parameter di `config.py` tidak berubah. Kalau parameter diubah (mis. `--nr-topics 15`), nomor topik berubah dan nama harus diisi ulang.

Draf nama saat ini (19 topik → 16 tema):

| Tema | Topik |
| --- | --- |
| KDM & bursa capres-cawapres 2029 | T1, T5 |
| Perdebatan & saling serang antarpendukung | T0 |
| Sindiran & candaan politik | T2 |
| Partai politik & sistem pemilu | T3 |
| Harapan & kriteria pemimpin bangsa | T4, T14 |
| Ekspresi religius | T6 |
| Pro-kontra Gibran | T7 |
| Dukungan singkat (ajakan & emoji) | T8 |
| Korupsi & penegakan hukum | T9 |
| Prabowo & Pramono | T10 |
| Isu ijazah & syarat pendidikan capres | T11 |
| Kepercayaan pada lembaga survei | T12 |
| Jokowi & keluarganya | T13 |
| Uang, anggaran & politik uang | T15 |
| Sikap terhadap PDIP | T16 |
| Kecurangan & hasil pemilu | T17 |
| Lainnya (penolakan singkat "no no no") | T18 |

### Label validasi sentimen

1. File: `data/processed/sentiment_label_sample.csv`, 400 komentar diambil acak berstrata (min. 10 per komunitas utama, sisanya dari small dan tanpa_interaksi).
2. Kolom `label_1` sudah diisi LLM (Claude); kolom `catatan` = "label oleh LLM (Claude)". Hasil: 169 negatif, 151 positif, 80 netral.
3. Aturan pelabelan:
    - positif: mendukung, memuji, berharap baik, setuju; termasuk "setuju", "betul", "aamiin", "yes", emoji hati.
    - negatif: menolak, mengkritik, menghina, menyindir; sarkasme dan ejekan beremoji tertawa dilabeli negatif.
    - netral: pertanyaan, prediksi/analisis tanpa sikap, ekspresi religius umum, maksud tidak jelas.
    - komentar campuran: sesuai sikap yang paling dominan.
4. Kalau ada label yang menurut Anda keliru, ubah langsung di `label_1`, lalu jalankan `python sentiment_analysis.py` lagi.
5. Membuat sampel baru (label lama hilang!): `python sentiment_analysis.py --make-sample --force`.

## 6. Kalau muncul error

| Gejala | Penyebab | Solusi |
| --- | --- | --- |
| `[BERTopic dilewati] Pustaka belum terpasang` | Pustaka NLP belum dipasang di venv | `pip install -r requirements-nlp.txt` |
| `OpenBLAS error: Memory allocation still failed` | Terlalu banyak thread/proses saat hitung koherensi | Sudah diperbaiki di kode; kalau masih muncul: `python topic_modeling.py --no-coherence` |
| Terminal macet, Ctrl+C tidak berfungsi | Proses Python hang setelah error | Tutup terminal (ikon tempat sampah di VS Code), atau `Stop-Process -Name python -Force`, atau Task Manager → End task Python |
| `ModuleNotFoundError` | venv belum aktif / pustaka belum ada | `venv\Scripts\activate`, lalu `pip install -r requirements.txt` |
| `'python' is not recognized` | Python tidak ada di PATH | Pakai `py nama_skrip.py` |
| `FileNotFoundError ... community_assignment.csv` | Tahap 5 belum dijalankan | `python detect_communities.py` dulu |
| gensim gagal dipasang (sering di Python 3.13) | Tidak ada wheel untuk versi Python itu | Lewati: `pip install bertopic sentence-transformers transformers torch`; koherensi saja yang kosong |
| torch gagal dipasang | Versi default terlalu besar/tidak cocok | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Nama topik manual tidak terbaca | File disimpan sebagai .xlsx | Simpan ulang sebagai CSV UTF-8 |

Waktu proses di laptop tanpa GPU: embedding BERTopic sekitar 1–2 menit (sekali, lalu di-cache), UMAP sekitar 2 menit tiap run, IndoBERT untuk ~7.000 komentar sekitar 5–15 menit.

## 7. Catatan untuk laporan dan sisa pekerjaan

### Sisa pekerjaan

- [ ] Jalankan `python topic_modeling.py` sekali lagi supaya nama tema dipakai di semua output.
- [ ] Jalankan `python sentiment_analysis.py` (IndoBERT + evaluasi terhadap 400 label).
- [ ] Jalankan `python compare_communities.py` untuk hasil final Tahap 9.
- [ ] Kalau F1 macro IndoBERT < 0,6, coba model lain dengan mengganti `SENTIMENT_MODEL` di `config.py` (mis. `hanifnoerr/Fine-tuned-Indonesian-Sentiment-Classifier`).
- [ ] Opsional: cek 20–30 label LLM secara acak.
- [ ] Perbarui angka lama di `docs/plan_pengerjaan_final_project.md` (masih mencatat data sebelum pengambilan ulang).

### Yang perlu ditulis di laporan

- **BAB II/III, closeness dan betweenness** dihitung pada graf tak berarah (LCC), bukan berarah, karena jaringan sangat jarang.
- **Batasan reply bertingkat**: reply dihubungkan ke penulis komentar utama; hubungan antar-reply direkonstruksi sebagian lewat @mention.
- **Arti komunitas**: komunitas sangat mengikuti video (NMI 0,84), jadi ditafsirkan sebagai kelompok diskusi per video/thread, bukan kelompok ideologis. Laporkan modularity LCC (0,88), bukan graf penuh (0,94).
- **Topik**: sebutkan parameter BERTopic (model embedding, min. 30 komentar, maks. 20 topik), koherensi c_v 0,39, outlier 27%, dan bahwa nama topik ditentukan manual. Pisahkan tema gaya berkomentar (perdebatan, sindiran, dukungan singkat) dari tema isu substantif.
- **Komunitas C12 dan C17** berisi diskusi keagamaan/Cak Nun, di luar isu capres.
- **Label validasi sentimen dibuat oleh LLM** (Claude) dengan pedoman tiga kelas; sebutkan di metodologi dan di keterbatasan (bukan label manusia, tanpa uji kesepakatan antarpelabel).
- **Disebut ≠ didukung**: frekuensi sebutan tokoh harus dibaca bersama sentimennya.
- **Kontrol efek video**: laporkan uji permutasi di dalam video di samping chi-square.
- **Bahasa daerah** (Jawa, Sunda) ikut masuk karena filter bahasa hanya di level video.
