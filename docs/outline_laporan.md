Berikut adalah usulan **Outline Laporan Final Project** yang diselaraskan dengan rumusan masalah, tujuan, dan metodologi/metode yang kita terapkan dalam proyek ini.

---

# OUTLINE LAPORAN FINAL PROJECT

## **BAB I: PENDAHULUAN**

### **1.1 Latar Belakang Masalah**
- **Dinamika Politik Digital Menuju Pilpres 2029:** Mendorong diskursus publik di platform media sosial, khususnya YouTube sebagai salah satu ruang diskusi politik utama di Indonesia.
- **Karakteristik Diskusi Komentar YouTube:** Percakapan publik menghasilkan data dalam jumlah besar (big data) yang kaya akan interaksi (*reply* & *@mention*), variasi isu, serta ekspresi opini/sentimen masyarakat.
- **Tantangan Analisis Manual:** Diskusi yang sangat cepat dan tersebar pada banyak video tidak memungkinkan dianalisis secara manual, sehingga membutuhkan pendekatan *Social Media Analytics*.
- **Integrasi Metode (SNA + Topic Modelling + Sentiment Analysis):**
  - *Social Network Analysis (SNA)* untuk memetakan struktur interaksi, mengidentifikasi aktor kunci, dan mendeteksi komunitas diskusi.
  - *Topic Modelling / Pemetaan Isu* untuk mengekstrak tema utama yang dibahas oleh tiap kelompok.
  - *Analisis Sentimen* untuk mengukur kecenderungan opini (positif, negatif, netral) di setiap komunitas.
- **Pentingnya Penelitian:** Memberikan pemahaman komprehensif mengenai bagaimana struktur jaringan sosial membentuk dan memengaruhi distribusi isu serta sentimen publik terkait spekulasi kandidat Pilpres 2029.

### **1.2 Rumusan Masalah**
1. Bagaimana pola jaringan interaksi dan komunitas yang terbentuk dalam diskusi komentar YouTube terkait spekulasi Pilpres 2029?
2. Siapa pengguna yang berperan penting dalam jaringan, dan isu apa yang dominan dibahas pada setiap komunitas?
3. Bagaimana distribusi sentimen positif, negatif, dan netral pada komentar di setiap komunitas?

### **1.3 Batasan Masalah**
1. **Sumber Data:** Data diambil terbatas dari platform YouTube melalui YouTube Data API v3 pada video bertema spekulasi Pilpres 2029 (5 kata kunci utama) dalam rentang publikasi 300 hari terakhir.
2. **Unit Interaksi:** *Edge* jaringan sosial dibentuk murni berdasarkan interaksi antar-pengguna (*reply* pada komentar induk dan *@mention* di dalam teks komentar), bukan sekadar komentar pada video yang sama (*co-commenting*).
3. **Anonimisasi Data:** Demi etika privasi data publik, identitas pengguna dianonimkan (`user_0001`, dsb.) dan analisis dilakukan pada level kelompok/komunitas, bukan afiliasi individu.
4. **Bahasa:** Analisis difokuskan pada komentar berbahasa Indonesia atau variasi slang/bahasa daerah di Indonesia.

### **1.4 Tujuan Penelitian**
1. Menganalisis dan memvisualisasikan struktur jaringan interaksi serta komunitas pengguna dalam diskusi komentar YouTube terkait spekulasi Pilpres 2029.
2. Mengidentifikasi pengguna yang berperan penting dalam jaringan serta menemukan isu dominan pada setiap komunitas.
3. Mengukur dan membandingkan distribusi sentimen positif, negatif, dan netral pada komentar di setiap komunitas.

### **1.5 Manfaat Penelitian**
1. **Manfaat Akademis:**
   - Memberikan kontribusi metodologis dalam penggabungan teknik *Social Network Analysis* (SNA), *Natural Language Processing* (NLP), dan analisis sentimen pada data diskusi bertingkat (*comment & reply*).
2. **Manfaat Praktis/Praktisi Media Sosial:**
   - Memberikan gambaran kepada peneliti media sosial, komunikator politik, atau pengambil kebijakan mengenai struktur percakapan publik, dinamika pembentukan komunitas, serta isu-isu sensitif terkait Pilpres 2029.

---

## **BAB II: TINJAUAN PUSTAKA**

### **2.1 Media Sosial YouTube sebagai Ruang Diskusi Politik**
- Peran YouTube sebagai media audio-visual dan kolom komentar sebagai ruang publik digital (*digital public sphere*).
- Karakteristik interaksi berantai (*threaded comments* & *reply*).

### **2.2 Text Mining dan Natural Language Processing (NLP) dalam Bahasa Indonesia**
- Konsep preprocessing teks berbahasa informal/slang pada media sosial.
- Tantangan analisis teks politik Indonesia (singkatan, kata slang, bahasa daerah, dan ekspresi emoji).

### **2.3 Analisis Jaringan Sosial (Social Network Analysis)**
- Definisi node (akun pengguna) dan edge (interaksi *reply* & *mention*).
- Graf berarah berbobot (*directed weighted graph*) vs. graf tak berarah.
- Konsep *Largest Connected Component* (LCC) dan keterfragmentasian jaringan (*sparsity*).

### **2.4 Metrik Sentralitas Jaringan (Network Centrality)**
- Peran sentralitas dalam mengidentifikasi aktor kunci (*influencer*, *broker*, *initiator*).

#### **2.4.1 Degree Centrality**
- In-degree (banyaknya balasan/mention yang diterima) dan Out-degree (banyaknya balasan/mention yang dikirim).

#### **2.4.2 Betweenness Centrality**
- Peran node sebagai jembatan (*broker*) pada jalur terpendek antar-kelompok.

#### **2.4.3 Closeness Centrality**
- Tingkat kedekatan/kecepatan jangkauan suatu node ke seluruh node lain dalam LCC.

#### **2.4.4 Eigenvector Centrality & PageRank**
- Sentralitas berbasis bobot dan kualitas koneksi (terhubung ke node lain yang berpengaruh).

### **2.5 Deteksi Komunitas (Community Detection)**
- Konsep modularity dalam jaringan sosial.
- Algoritma Louvain dan Leiden untuk pengelompokan node.
- Uji kestabilan hasil (*seed stability*) dan penentuan parameter *resolution*.

### **2.6 Pemetaan Isu (Topic Modelling)**
- Ekstraksi topik dari komentar kelompok menggunakan pendekatan NLP (misal: BERTopic / TF-IDF / LDA).

### **2.7 Analisis Sentimen**
- Klasifikasi opini (Positif, Negatif, Netral) menggunakan model *pre-trained* Transformer (IndoBERT/IndoRoBERTa).
- Pengujian statistik hubungan antar-komunitas dan distribusi sentimen (uji *Chi-Square* dan *Cramér's V*).

---

## **BAB III: METODOLOGI PENELITIAN**

### **3.1 Alur Penelitian**
- Diagram alir penelitian (Pengumpulan Data $\rightarrow$ Preprocessing $\rightarrow$ Pembuatan Graf & SNA $\rightarrow$ Community Detection $\rightarrow$ Pemetaan Isu $\rightarrow$ Analisis Sentimen $\rightarrow$ Perbandingan Antar-Komunitas).

### **3.2 Pengumpulan Data (YouTube Data API v3)**
- Kriteria seleksi video (5 kata kunci, rentang publikasi 300 hari, kriteria *min-comments*).
- Pengambilan komentar tingkat atas (`commentThreads.list`) dan balasan lengkap (`comments.list` via `parentId`).

### **3.3 Preprocessing Data**
- **Data Jaringan:** Anonimisasi akun (`user_0001`), pembersihan zero-width character, penanganan akun tanpa nama.
- **Data Teks:**
  - Versi *Clean Light* (`text_light`): konversi emoji ke kata, normalisasi tanda baca & perpanjangan huruf (untuk Analisis Sentimen).
  - Versi *Clean Full* (`text_full`): *case folding*, eliminasi URL, normalisasi slang/singkatan, eliminasi *stopwords* & *laughter* (untuk Topic Modelling).
- **Flagging Kualitas:** Penandaan komentar kosong, pendek, spam, dan *copy-paste*.

### **3.4 Membangun Graf Interaksi dan Analisis Jaringan**
- Konstruksi graf berarah berbobot (`reply` dan `mention`).
- Pemisahan *self-loop*.
- Perhitungan metrik jaringan (Density, LCC, Clustering, Assortativity, Power-law degree fit) dan centrality (Degree, Betweenness, Closeness, PageRank, Eigenvector).
- Penentuan dan pemetaan **Akun Penjembatan** (*Bridging Accounts*) antar-video.

### **3.5 Deteksi Komunitas dan Evaluasi**
- Konversi graf ke tak berarah berbobot.
- Penerapan algoritma Leiden & Louvain, *resolution sweep*, dan evaluasi medoid dari 20 *seed*.
- Pengelompokan komunitas minimum (`COMMUNITY_MIN_SIZE`) dan label kelompok `"small"`.
- Evaluasi keterkaitan komunitas dengan video (*Normalized Mutual Information* / NMI).

### **3.6 Penggabungan Komentar dengan Komunitas (Join)**
- Menggabungkan data komentar dengan ID komunitas penulisnya (`comments_with_community.csv`).
- Pemisahan kelompok pembanding `"tanpa_interaksi"` untuk akun yang tidak masuk graf jaringan.

### **3.7 Pemetaan Isu per Komunitas**
- Ekstraksi kata kunci khas dan topik dominan pada tiap komunitas utama.

### **3.8 Analisis Sentimen dan Uji Statistik**
- Klasifikasi sentimen pada level komentar.
- Evaluasi akurasi model pada sampel berlabel manual.
- Agregasi proporsi sentimen per komunitas dan uji hipotesis (*Chi-Square test of independence*).

---

## **BAB IV: HASIL DAN PEMBAHASAN**

### **4.1 Deskripsi Dataset dan Hasil Pengumpulan Data**
- Statistik dataset mentah: Jumlah video unik, jumlah komentar tingkat atas, jumlah *reply*, dan total akun unik.

### **4.2 Hasil Preprocessing Data**
- Ringkasan hasil pembersihan teks, eliminasi duplikat, dan distribusi flag kualitas (`keep_sentiment` & `keep_topic`).

### **4.3 Hasil Analisis Jaringan Sosial (SNA)**
- **Topologi Jaringan:** Jumlah node, edge, kerapatan (*density*), ukuran LCC, serta distribusi degree (*power-law fit*).
- **Aktor Kunci (Centrality):** Pembahasan top-N aktor per metrik centrality (siapa penggerak percakapan, pembawa respon terbanyak, dan penghubung antar-kelompok).
- **Peran Akun Penjembatan (*Bridging Accounts*):** Analisis akun yang menghubungkan diskusi lintas video.

### **4.4 Hasil Deteksi Komunitas**
- Perbandingan performa Louvain vs Leiden, variasi *resolution*, dan kestabilan *seed*.
- Karakteristik komunitas utama yang terbentuk (ukuran, *density* internal, *hub* utama).
- Evaluasi derajat keterikatan komunitas terhadap spesifisitas video (*Video Association / NMI*).

### **4.5 Hasil Pemetaan Isu per Komunitas**
- Isu, tokoh politik, dan kata kunci dominan yang muncul pada masing-masing komunitas diskusi.

### **4.6 Hasil Analisis Sentimen dan Perbandingan Antar-Komunitas**
- Distribusi sentimen global (Positif, Negatif, Netral).
- Perbandingan proporsi sentimen antar-komunitas utama vs kelompok `"tanpa_interaksi"`.
- Hasil uji statistik *Chi-Square* dan analisis *effect size* (Cramér's V) mengenai perbedaan pola sentimen antar-komunitas.

### **4.7 Pembahasan dan Diskusi Implikasi**
- Sintesis keterhubungan antara struktur jaringan (SNA), pemetaan isu, dan preferensi sentimen politik publik.
- Implikasi praktis dan analisis keterbatasan penelitian (*limitations*).

---

## **BAB V: KESIMPULAN DAN SARAN**

### **5.1 Kesimpulan**
1. **Struktur Jaringan & Komunitas:** Jawaban ringkas atas Rumusan Masalah 1 mengenai bentuk jaringan sosial, LCC, dan komunitas utama yang terbentuk.
2. **Aktor Kunci & Pemetaan Isu:** Jawaban ringkas atas Rumusan Masalah 2 mengenai aktor sentral/penjembatan dan isu utama pada tiap komunitas.
3. **Distribusi Sentimen Antar-Komunitas:** Jawaban ringkas atas Rumusan Masalah 3 mengenai kecenderungan sentimen dan signifikansi perbedaannya antar-komunitas.

### **5.2 Saran**
- **Saran Metodologis:** Pengembangan jaringan pada platform lain atau eksplorasi model NLP/LLM berbahasa Indonesia yang lebih kuat terhadap sarkasme politik.
- **Saran Pengambilan Data:** Perluasan sampel video atau analisis berdasarkan garis waktu (*time-series*) untuk melihat pergeseran isu menjelang 2029.
