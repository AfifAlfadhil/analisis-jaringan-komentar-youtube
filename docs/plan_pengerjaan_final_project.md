# Rencana Pengerjaan Final Project

**Judul:** Analisis Jaringan Interaksi Publik, Pemetaan Isu, dan Analisis Sentimen Terkait Spekulasi Pilpres 2029 pada Diskusi Komentar YouTube

---

## 1. Ringkasan

| Aspek | Keterangan |
|---|---|
| Sumber data | Komentar dan balasan (reply) pada video YouTube bertema Pilpres 2029 |
| Metode utama | Social Network Analysis (SNA), community detection, topic modelling, analisis sentimen |
| Unit analisis | Akun = node jaringan; komentar = unit analisis isu dan sentimen |
| Luaran | Jaringan interaksi, daftar komunitas, peta isu per komunitas, perbandingan sentimen antar komunitas, laporan, dan presentasi |

### Pertanyaan penelitian

1. Bagaimana struktur jaringan interaksi antar-pengguna dalam diskusi komentar? Siapa aktor yang paling berperan?
2. Komunitas diskusi apa saja yang terbentuk dari pola reply?
3. Isu dan kata kunci apa yang dominan di tiap komunitas?
4. Bagaimana kecenderungan sentimen (positif, negatif, netral) antar komunitas?

---

## 2. Alur Besar

```
Pengumpulan data (YouTube Data API)
        |
        v
Preprocessing (video + komentar)
        |
        +-------------------------+
        |                         |
        v                         v
Bangun graf akun            Analisis komentar
(edge = reply/mention)      (topik + sentimen per komentar)
        |                         |
        v                         |
SNA + centrality                  |
        |                         |
        v                         |
Community detection               |
(tabel: akun -> komunitas)        |
        |                         |
        +------------+------------+
                     |
                     v
       Join: komentar mewarisi komunitas penulisnya
                     |
                     v
   Agregasi per komunitas (isu + sentimen) + uji statistik
                     |
                     v
         Visualisasi, laporan, presentasi
```

Catatan desain: akun yang menulis banyak komentar dengan isu dan sentimen beragam **tidak perlu diberi satu label isu atau sentimen**. Label melekat pada komentar, akun hanya meneruskan label komunitas ke komentarnya.

---

## 3. Tahapan Pengerjaan

### Tahap 0: Persiapan

- [ ] Buat project di Google Cloud Console, aktifkan YouTube Data API v3, buat API key
- [ ] Siapkan environment Python (venv/conda) dan repo (struktur di bagian 6)
- [ ] Install library: `google-api-python-client`, `pandas`, `networkx`, `python-louvain` / `igraph` + `leidenalg`, `bertopic`, `transformers`, `torch`, `scikit-learn`, `matplotlib`, `plotly`
- [ ] Tetapkan definisi operasional: apa itu "edge", apa itu "komunitas", rentang waktu data, bahasa yang dianalisis
- [ ] Perbaiki judul (typo "Pillres" menjadi "Pilpres")

### Tahap 1: Pengumpulan data

**Video**
- [x] Cari video dengan 5 kata kunci: *Pilpres 2029, Kandidat Presiden 2029, Calon Presiden 2029, Prediksi Pilpres 2029, Bursa Capres 2029*
- [x] Tetapkan kriteria seleksi video: rentang tanggal publikasi (120 hari terakhir), jumlah komentar minimum (50), bahasa Indonesia, komentar tidak dimatikan (parameter di `config.py`)
- [x] Simpan metadata: judul, channel, tanggal publikasi, views, likes, jumlah komentar, deskripsi, kata kunci penemu

**Komentar**
- [x] Ambil komentar tingkat atas lewat `commentThreads.list`
- [x] Ambil balasan lewat `comments.list` (parameter `parentId`) untuk thread yang punya reply
- [x] Simpan: ID komentar, ID kanal + nama penulis, isi, waktu, jumlah like, jumlah reply, `parent_id`, `video_id`

**Hal yang perlu diperhatikan**
- Kuota default API sekitar 10.000 unit per hari. `search.list` mahal (100 unit per panggilan), sedangkan `commentThreads.list`/`comments.list` murah (1 unit). Rencanakan pengambilan beberapa hari dan simpan hasil mentah ke file supaya tidak perlu mengulang.
- Balasan di YouTube hanya satu tingkat di bawah komentar tingkat atas. Balasan yang menyebut `@nama` pengguna lain bisa dipakai sebagai sumber edge tambahan.
- Catat tanggal pengambilan data (jumlah komentar bisa berubah).

**Output:** `data/raw/videos.csv`, `data/raw/comments.csv` (dibuat oleh `python main.py`)

**Hasil pengambilan (3 Nov 2026):** 36 video unik, 8.518 komentar dan reply (5.494 tingkat atas, 3.024 reply), 6.730 akun unik. Kuota terpakai 678 unit. Batas 300 thread per video tercapai pada 10 video, sisanya diambil lengkap.

### Tahap 2: Preprocessing

**Data jaringan (level akun)**
- [x] Hapus duplikat komentar
- [x] Standarkan ID akun (memakai `authorChannelId`; nama tampilan hanya untuk mencocokkan `@mention`)
- [x] Anonimisasi username untuk laporan dan visualisasi (mis. `user_0001`; pemetaan di `data/processed/author_map.csv`, privat)
- [x] Tangani akun terhapus atau tidak bernama

**Data teks (level komentar)**
- [x] Case folding, hapus URL, normalisasi mention dan emoji (emoji bisa dikonversi ke teks karena membawa sinyal sentimen)
- [x] Normalisasi kata tidak baku/slang (mis. kamus slang Indonesia) — kamus awal di `src/lexicons.py`, perluas setelah eksplorasi Tahap 7
- [x] Hapus stopword (untuk topic modelling saja, **jangan** untuk model sentimen berbasis transformer)
- [x] Filter komentar terlalu pendek atau tidak informatif (berupa flag `is_short`, `keep_topic`, `keep_sentiment`)
- [x] Deteksi dan tangani spam atau bot sederhana (komentar identik berulang, tautan promosi) — hasil: 0 spam, 0 copy-paste pada data saat ini
- [x] Buat dua versi teks: *bersih ringan* (`text_light`, untuk model sentimen) dan *bersih penuh* (`text_full`, untuk topic modelling)

**Output:** `data/processed/comments_clean.csv` (dibuat oleh `python preprocess.py`)

### Tahap 3: Membangun graf interaksi

- [x] Definisikan edge: **penulis reply → penulis komentar induk** (directed)
- [x] Tambahkan edge dari mention `@nama` di dalam teks (opsional, tandai tipe edge-nya)
- [x] Bobot edge = jumlah interaksi antar pasangan akun (bukan edge ganda)
- [x] Hapus self-loop (akun membalas dirinya sendiri), atau catat terpisah (`edges_selfloop.csv`)
- [x] Simpan graf (`.graphml` / `.gexf` untuk Gephi, dan edge list CSV)

**Output:** `data/processed/edges.csv`, `outputs/graph/network.graphml` (dibuat oleh `python build_graph.py`)

### Tahap 4: Analisis jaringan (SNA)

**Statistik jaringan**
- [x] Jumlah node dan edge, density, jumlah komponen terhubung, ukuran largest connected component (LCC)
- [x] Rata-rata degree, clustering coefficient, diameter/average path length (pada LCC)
- [x] Distribusi degree (cek apakah mendekati power law)

**Centrality (tiap metrik dijelaskan perannya)**
- [x] Degree / in-degree / out-degree: siapa yang paling banyak berinteraksi, dibalas, atau membalas
- [x] Betweenness: penghubung antar-kelompok
- [x] Closeness: kedekatan ke seluruh jaringan
- [x] Eigenvector atau PageRank: pengaruh berdasarkan koneksi ke aktor penting

**Output:** `outputs/tables/network_stats.csv`, `outputs/tables/centrality.csv`, `outputs/tables/top_actors.csv`, `outputs/figures/degree_distribution.png` (dibuat oleh `python analyze_network.py`)

**Hasil (graf 2.601 node, 3.139 edge):** LCC 1.835 node (70,6%); 766 node di luar LCC tidak ikut analisis jarak/closeness/eigenvector. Avg degree 2,30, median 1; clustering 0,18; transitivity 0,02; reciprocity 0,09; assortativity negatif (-0,13). Pada LCC: avg path length 7,35, diameter 21. Distribusi degree heavy-tailed (power-law alpha 2,49, xmin 3, KS 0,02; indikasi saja, belum diuji terhadap log-normal).

> Karena jaringan YouTube cenderung jarang (*sparse*) dan terfragmentasi, analisis lanjutan (komunitas, path length) difokuskan pada LCC atau pada komponen yang cukup besar. Tulis jumlah node yang tersisa dan yang dibuang.

### Tahap 5: Community detection

- [x] Ubah graf ke undirected berbobot untuk algoritma yang memerlukannya
- [x] Jalankan Louvain (baseline) dan Leiden (pembanding)
- [x] Ukur modularity; jalankan berulang dengan seed berbeda dan cek kestabilan hasil
- [x] Coba beberapa nilai resolution, pilih yang interpretatif
- [x] Tetapkan batas ukuran komunitas minimum yang dianalisis (mis. 10 akun) dan gabungkan sisanya sebagai "komunitas kecil"
- [x] Buat tabel `akun -> id_komunitas`

**Output:** `outputs/tables/community_assignment.csv`, `community_stats.csv` (ukuran, density internal, hub utama, video dominan), `community_sweep.csv`, `community_summary.csv` (dibuat oleh `python detect_communities.py`; parameter di `config.py`)

**Hasil (Leiden, resolution 1,0, medoid dari 20 seed):**
- Modularity 0,923 pada graf penuh dan 0,895 pada LCC saja (nilai graf penuh terlalu tinggi karena ratusan komponen kecil otomatis menjadi komunitas, jadi laporkan angka LCC). Louvain 0,922; ARI Louvain vs Leiden 0,96; ARI antar-seed rata-rata 0,98.
- Resolution 0,5 sampai 2,0 menghasilkan modularity hampir sama (0,919-0,923); 1,0 dipilih karena modularity tertinggi dan nilai standar.
- 29 komunitas berukuran >= 10 akun (1.888 akun), 713 akun di "small". Tanpa 7 pemilik channel: 27 komunitas, ARI 0,95 terhadap hasil utama (hasil tidak bergantung pada mereka).
- **Kaitan dengan video:** NMI komunitas vs video utama akun = 0,74 (pada akun di komunitas >= 10). Rata-rata pangsa video dominan per komunitas 0,91; 12 dari 29 komunitas hanya berasal dari satu video dan 7 komunitas lintas-video (pangsa video dominan < 0,8). Komunitas sebagian besar mencerminkan diskusi per video/thread; tafsirkan sebagai "kelompok diskusi", bukan kelompok ideologis.

### Tahap 6: Menggabungkan komentar dengan komunitas

- [ ] `join` tabel komentar dengan tabel komunitas lewat ID akun
- [ ] Beri label **"tanpa interaksi"** untuk akun yang tidak masuk jaringan (isolat), sebagai kelompok pembanding
- [ ] Laporkan: jumlah komentar yang masuk komunitas, yang di kelompok tanpa interaksi, dan yang dibuang

**Output:** `data/processed/comments_with_community.csv`

### Tahap 7: Analisis isu

- [ ] Eksplorasi awal: frekuensi kata, bigram/trigram, word cloud kasar
- [ ] Topic modelling di level komentar: BERTopic (embedding bahasa Indonesia/multilingual) dan/atau LDA sebagai pembanding
- [ ] Evaluasi: topic coherence, pengecekan manual contoh komentar per topik
- [ ] Beri nama topik secara manual berdasarkan kata kunci dan contoh komentar
- [ ] Hitung distribusi topik **per komunitas** dan kata kunci khas tiap komunitas (mis. TF-IDF antar-komunitas)
- [ ] Sebut tokoh/entitas politik yang paling sering muncul (named entity atau kamus nama tokoh)

**Output:** `outputs/tables/topics.csv`, `outputs/tables/topic_by_community.csv`

### Tahap 8: Analisis sentimen

- [ ] Tentukan pendekatan: model pre-trained bahasa Indonesia (mis. IndoBERT/IndoRoBERTa yang tersedia di Hugging Face, cek ketersediaan dan lisensinya), atau lexicon-based sebagai baseline
- [ ] Buat **sampel berlabel manual** (mis. 300 sampai 500 komentar, acak berstrata per komunitas) untuk validasi
- [ ] Jika memungkinkan, dua orang pelabel dan hitung Cohen's kappa
- [ ] Evaluasi: accuracy, precision, recall, F1 (macro). Bandingkan model dengan baseline
- [ ] Jalankan model pada seluruh komentar, simpan label dan skor/probabilitas
- [ ] Catat keterbatasan: sarkasme, bahasa campur, singkatan, konteks politik

**Output:** `outputs/tables/sentiment_predictions.csv`, `outputs/tables/sentiment_eval.csv`

### Tahap 9: Perbandingan antar komunitas

- [ ] Hitung proporsi sentimen per komunitas dalam dua versi: **berbasis komentar** dan **berbasis akun** (rata-rata per akun dulu, baru antar-akun), untuk mengecek apakah akun yang sangat aktif mendominasi
- [ ] Uji hubungan komunitas dan sentimen (chi-square, plus effect size seperti Cramér's V)
- [ ] Hubungkan hasil: topik apa yang dominan di komunitas yang cenderung positif/negatif
- [ ] Bandingkan komunitas dengan kelompok "tanpa interaksi"
- [ ] Tulis temuan sebagai interpretasi deskriptif, hindari klaim kausal

### Tahap 10: Visualisasi dan dashboard

- [ ] Graf jaringan berwarna per komunitas, ukuran node = centrality (Gephi atau `networkx`/`pyvis`)
- [ ] Bar chart top aktor menurut centrality
- [ ] Bar bertumpuk proporsi sentimen per komunitas
- [ ] Peta topik: kata kunci atau heatmap topik x komunitas
- [ ] Tren waktu komentar dan sentimen (opsional)
- [ ] Dashboard ringkas (Streamlit/Plotly) bila dibutuhkan

### Tahap 11: Laporan dan presentasi

- [ ] Susun laporan: pendahuluan, tinjauan pustaka, metodologi, hasil, pembahasan, batasan, kesimpulan
- [ ] Susun slide presentasi (alur: masalah, data, metode, temuan jaringan, temuan isu, temuan sentimen, kesimpulan)
- [ ] Rapikan kode dan buat README agar bisa direproduksi

---

## 4. Timeline Usulan

Sesuaikan dengan tenggat dari dosen. Contoh kerangka 6 minggu:

| Minggu | Fokus | Hasil |
|---|---|---|
| 1 | Persiapan, seleksi video, pengambilan data | Dataset mentah video + komentar |
| 2 | Preprocessing, bangun graf, SNA dasar | Edge list, statistik jaringan, centrality |
| 3 | Community detection, join komentar-komunitas, eksplorasi teks | Tabel komunitas, dataset gabungan |
| 4 | Topic modelling dan analisis isu per komunitas | Topik, kata kunci per komunitas |
| 5 | Analisis sentimen (pelabelan sampel, model, evaluasi) dan perbandingan antar komunitas | Sentimen per komunitas, uji statistik |
| 6 | Visualisasi, laporan, slide, latihan presentasi | Laporan dan presentasi final |

Beri cadangan waktu di minggu 1 (kuota API dan perubahan kriteria video biasanya memakan waktu) dan di minggu 5 (pelabelan manual).

---

## 5. Tools

| Kebutuhan | Pilihan |
|---|---|
| Pengambilan data | YouTube Data API v3 (`google-api-python-client`) |
| Olah data | `pandas`, `numpy` |
| Jaringan | `networkx`, `igraph`, `python-louvain`, `leidenalg`, Gephi |
| Topic modelling | `bertopic`, `gensim` (LDA), `scikit-learn` |
| Sentimen | `transformers` (IndoBERT/IndoRoBERTa), lexicon sebagai baseline |
| Visualisasi | `matplotlib`, `plotly`, `pyvis`, Gephi |
| Dashboard (opsional) | Streamlit |

---

## 6. Struktur Folder Repo

```
project/
├── README.md
├── requirements.txt
├── config.yaml                 # keyword, kriteria video, parameter
├── data/
│   ├── raw/                    # hasil API, jangan diubah
│   └── processed/              # data bersih, edges, komentar + komunitas
├── notebooks/
│   ├── 01_collect.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_network_sna.ipynb
│   ├── 04_community.ipynb
│   ├── 05_topics.ipynb
│   ├── 06_sentiment.ipynb
│   └── 07_compare_visualize.ipynb
├── src/                        # fungsi yang dipakai ulang
└── outputs/
    ├── graph/
    ├── tables/
    └── figures/
```

---

## 7. Skema Data

**videos**

| Kolom | Keterangan |
|---|---|
| video_id | ID video |
| title, channel, published_at | Judul, nama channel, tanggal |
| views, likes, comment_count | Metrik video |
| description | Deskripsi video |
| keyword | Kata kunci pencarian yang menemukan video |

**comments**

| Kolom | Keterangan |
|---|---|
| comment_id | ID komentar |
| video_id | Relasi ke video |
| parent_id | Kosong jika komentar tingkat atas; berisi ID komentar induk jika reply |
| author_id | ID kanal penulis (anonimkan untuk laporan) |
| text | Isi komentar |
| published_at | Waktu komentar |
| like_count, reply_count | Metrik komentar |

**edges**

| Kolom | Keterangan |
|---|---|
| source | Akun yang membalas atau me-mention |
| target | Akun yang dibalas atau di-mention |
| weight | Jumlah interaksi |
| type | `reply` atau `mention` |

**comments_with_community** (hasil gabungan)

| Kolom | Keterangan |
|---|---|
| comment_id, author_id, text | Dari tabel komentar |
| community_id | Dari hasil community detection (atau "tanpa interaksi") |
| topic_id | Dari topic modelling |
| sentiment, sentiment_score | Dari model sentimen |

---

## 8. Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Jaringan sparse dan terfragmentasi | Komunitas sedikit dan kecil, hasil kurang bermakna | Tambah jumlah video, gunakan edge mention, fokus pada LCC, jelaskan di batasan |
| Kuota API habis | Pengambilan data tersendat | Rencanakan per hari, simpan hasil bertahap, kurangi `search.list` |
| Balasan hanya satu tingkat | Struktur percakapan dangkal | Dokumentasikan; gunakan mention sebagai edge tambahan |
| Akun sangat aktif mendominasi komunitas | Sentimen komunitas bias | Laporkan versi berbasis komentar dan berbasis akun |
| Sentimen salah klasifikasi (sarkasme, slang) | Akurasi turun | Validasi dengan sampel manual, laporkan F1 dan contoh kesalahan |
| Bot/spam | Mengganggu jaringan dan topik | Filter komentar duplikat dan tautan promosi |
| Hasil community detection tidak stabil | Interpretasi lemah | Jalankan berulang, bandingkan algoritma, laporkan modularity |
| Isu politik yang sensitif | Etika dan privasi | Anonimkan akun, jangan tampilkan username asli, tulis temuan secara deskriptif |

---

## 9. Etika dan Privasi

- Data publik, tetapi tetap **anonimkan** akun di laporan, slide, dan visualisasi
- Jangan menyimpulkan afiliasi politik individu dari komentar; analisis berada di level kelompok
- Sebutkan batasan: sampel video bukan representasi seluruh masyarakat, pengguna YouTube tidak mewakili populasi pemilih
- Patuhi ketentuan layanan YouTube API

---

## 10. Checklist Deliverables

- [ ] Dataset video dan komentar (mentah + bersih)
- [ ] Graf interaksi dan edge list
- [ ] Tabel centrality dan daftar aktor kunci
- [ ] Tabel komunitas dan statistik per komunitas
- [ ] Hasil topic modelling dan peta isu per komunitas
- [ ] Hasil sentimen, evaluasi model, dan sampel berlabel manual
- [ ] Perbandingan sentimen dan isu antar komunitas (plus uji statistik)
- [ ] Visualisasi final (jaringan, bar, heatmap)
- [ ] Laporan final
- [ ] Slide presentasi
- [ ] Repo yang rapi dan README
