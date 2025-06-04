# Laporan Proyek Machine Learning - Hanan Naufal Indratma

## Project Overview

Di era digital saat ini, volume informasi yang tersedia sangat besar, menyebabkan tantangan dalam menemukan konten yang relevan bagi pengguna. Sistem rekomendasi menjadi solusi penting untuk menyaring informasi ini dan menyediakan pengalaman yang dipersonalisasi. Dalam industri hiburan, seperti layanan streaming dan e-commerce, sistem rekomendasi telah terbukti meningkatkan keterlibatan pengguna dan pendapatan perusahaan. Sebagai contoh, Netflix melaporkan bahwa sistem rekomendasinya bertanggung jawab atas sekitar 75% dari aktivitas penayangan pengguna [1].

Namun, tantangan seperti *cold start* (ketika data pengguna atau item baru sangat sedikit) dan *data sparsity* (ketika interaksi pengguna-item jarang) masih menjadi hambatan dalam pengembangan sistem rekomendasi yang efektif. Oleh karena itu, penting untuk mengeksplorasi dan membandingkan berbagai pendekatan sistem rekomendasi untuk mengatasi masalah ini.

Studi sebelumnya menunjukkan bahwa integrasi antara pendekatan content-based dan collaborative filtering dapat meningkatkan akurasi serta kepuasan pengguna terhadap hasil rekomendasi [2]. Oleh karena itu, memahami kekuatan dan keterbatasan masing-masing pendekatan menjadi langkah penting menuju pengembangan sistem rekomendasi yang lebih optimal.

Referensi:

[1] A. Gomez-Uribe dan N. Hunt, "The Netflix Recommender System: Algorithms, Business Value, and Innovation," *ACM Transactions on Management Information Systems*, vol. 6, no. 4, pp. 1–19, Dec. 2016.

[2] J. A. Smith dan M. K. Johnson, "An Integrated Approach to Movie Recommendation: Collaborative Filtering and Content-Based Filtering Fusion," *International Journal of Creative Research Thoughts (IJCRT)*, vol. 12, no. 2, pp. 962–970, Feb. 2024.

## Business Understanding

Untuk membangun sistem rekomendasi yang efektif, pemahaman terhadap kebutuhan bisnis dan pengguna akhir sangat penting. Proses klarifikasi masalah dilakukan untuk merumuskan tantangan utama yang dihadapi dalam bentuk pernyataan masalah yang terukur, sekaligus menetapkan tujuan proyek yang relevan dengan konteks data dan kebutuhan pengguna.

### Problem Statements

- **Pernyataan Masalah 1**: Pengguna sering kesulitan menemukan film yang relevan atau sesuai preferensi pribadi karena jumlah film yang sangat banyak.
- **Pernyataan Masalah 2**: Sistem rekomendasi yang ada terkadang tidak akurat atau repetitif, hanya menampilkan film-film populer tanpa mempertimbangkan minat spesifik pengguna.
- **Pernyataan Masalah 3**: Kurangnya sistem rekomendasi yang mampu menggabungkan konteks konten film dan pola perilaku pengguna dalam satu pendekatan yang terpadu.

### Goals

- **Goal 1**: Membantu pengguna menemukan film yang sesuai dengan preferensi personal mereka melalui sistem rekomendasi yang cerdas.
- **Goal 2**: Meningkatkan kualitas dan relevansi rekomendasi dengan mempertimbangkan tidak hanya popularitas tetapi juga kesamaan konten dan perilaku pengguna serupa.
- **Goal 3**: Mengembangkan pendekatan sistem rekomendasi berbasis machine learning yang dapat mengatasi masalah cold start dan data sparsity secara efektif.

### Solution Approach

Untuk mencapai tujuan di atas, dua pendekatan utama dalam sistem rekomendasi digunakan dan dianalisis:

#### Solution 1: Content-Based Filtering
Pendekatan ini merekomendasikan film kepada pengguna berdasarkan kemiripan karakteristik konten film yang sebelumnya disukai pengguna. Fitur seperti genre, dan tag digunakan untuk mengukur kedekatan antar item. Model ini bersifat personal dan independen dari preferensi pengguna lain, sehingga efektif untuk pengguna baru (*cold-start users*) yang memiliki sedikit interaksi historis.

#### Solution 2: Collaborative Filtering
Pendekatan ini merekomendasikan film berdasarkan kesamaan perilaku antara pengguna. Jika dua pengguna memberikan rating yang serupa pada sejumlah film, maka mereka dianggap memiliki preferensi yang mirip. Dalam proyek ini digunakan pendekatan **model-based collaborative filtering** menggunakan embedding layer berbasis neural network (RecommenderNet) yang mempelajari representasi laten dari pengguna dan film untuk menghasilkan prediksi rating.

Melalui dua pendekatan ini, sistem diharapkan dapat memberikan rekomendasi yang lebih **personal, relevan, dan bervariasi** dengan memanfaatkan baik fitur konten film maupun perilaku kolektif pengguna.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah [MovieLens Dataset](https://www.kaggle.com/datasets/aigamer/movie-lens-dataset?select=tags.csv), yang disediakan oleh GroupLens Research. Dataset ini berisi data interaksi antara pengguna dan film, termasuk penilaian (rating), tag, metadata film, dan informasi penghubung ke sumber eksternal seperti IMDb dan TMDb dengan jumlah total interaksi (ratings) sebanyak **163,046**.

### Informasi Umum Dataset

Dataset terdiri dari beberapa file utama:

- `ratings.csv`: Data interaksi pengguna dengan film dalam bentuk rating.
- `movies.csv`: Informasi metadata film, termasuk judul dan genre.
- `tags.csv`: Informasi tambahan berupa tag yang diberikan oleh pengguna pada film tertentu.
- `links.csv`: Penghubung antara `movieId` dengan ID di IMDb dan TMDb.

### Variabel pada Dataset

#### ratings.csv

- `userId`: ID unik untuk masing-masing pengguna.
- `movieId`: ID unik untuk masing-masing film.
- `rating`: Penilaian yang diberikan pengguna terhadap film dalam skala 0.5–5.0.
- `timestamp`: Waktu saat rating diberikan (format Unix timestamp).

#### movies.csv

- `movieId`: ID unik untuk masing-masing film.
- `title`: Judul film beserta tahun rilis.
- `genres`: Daftar genre dari film (dipisahkan tanda `|` jika lebih dari satu).

#### tags.csv

- `userId`: ID unik pengguna.
- `movieId`: ID unik film.
- `tag`: Tag yang diberikan pengguna pada film.
- `timestamp`: Waktu saat tag diberikan (format Unix timestamp).

#### links.csv

- `movieId`: ID unik film dalam sistem MovieLens.
- `imdbId`: ID film dalam basis data IMDb.
- `tmdbId`: ID film dalam basis data TMDb.

### Exploratory Data Analysis (EDA)

Sebelum ekplorasi dilakukan, **dilakukan merge** pada keempat dataset. Kemudian, beberapa tahapan eksplorasi data dilakukan untuk memahami struktur dan kualitas data:

1. **Cek Tipe Data**
   Semua kolom pada file utama memiliki tipe data yang sesuai untuk analisis dan pemodelan.
    ```python
    df.info()
    ```
    Output:
    | No | Column                   | Non-Null Count | Dtype   |
    |----|--------------------------|----------------|---------|
    | 0  | userId                   | 3476           | int64   |
    | 1  | movieId                  | 3476           | int64   |
    | 2  | rating                   | 3476           | float64 |
    | 3  | title                    | 3476           | object  |
    | 4  | genres                   | 3476           | object  |
    | 5  | tag                      | 3476           | object  |
   
    Berdasarkan pengecekan pada tipe data, diperoleh tipe data userId dan movieId adalah integer, rating adalah float, serta variabel lain memiliki tipe data object.

3. **Cek Missing Value**  
    ```python
    df.isnull().sum().sum()
    ```
    Output:
    ```np.int64(0)``` 

    Tidak ditemukan *missing value* pada data.

4. **Cek Duplikasi**  
    ```python
    df.duplicated().sum()
    ```
    Output:
    ```np.int64(0)```
   Tidak ditemukan baris duplikat dalam data rating, sehingga data dapat langsung digunakan untuk training model.

5. **Cek Dimensi Data**  
    ```python
    df.shape
    ```
    Output:
    ```(3476, 6)``` 
    Data memiliki sebanyak 3476 baris dan 6 kolom.

6. **Jumlah Unik User dan Movie**  
    ```python
    df['userId'].nunique()
    df['movieId'].nunique()
    ```
    
    Output:
    ```54```
    ```1464``` 
   - User unik: **54 pengguna**
   - Film unik: **1464 film**

## Data Preparation
Pada tahap ini, dilakukan serangkaian proses untuk mempersiapkan data sebelum masuk ke tahap pemodelan. Terdapat dua pendekatan yang digunakan dalam proyek ini, yaitu **Content-Based Filtering** dan **Collaborative Filtering**, yang masing-masing memerlukan proses **persiapan data yang berbeda**.

### Content-Based Filtering

#### 1. Pembersihan Variabel genres
- Mengganti nilai `"(no genres listed)"` dengan `"None"` agar lebih mudah dalam pemrosesan teks.
- Mengubah nilai dalam kolom `genres` dari string yang dipisahkan oleh simbol `|` menjadi list Python.  
- **Alasan:** Format list memudahkan penggabungan genre dengan tag, serta meningkatkan efektivitas pemrosesan fitur berbasis teks.

#### 2. Pembersihan Variabel tags
- Menggabungkan semua tag berdasarkan `movieId` ke dalam satu baris, lalu mengubahnya menjadi format list.
- Menghilangkan nilai kosong atau null tag setelah proses agregasi.
- **Alasan:** Proses ini menyatukan informasi tag agar setiap film hanya memiliki satu baris representasi deskriptif.

#### 3. Mengatasi Missing Values
- Setelah proses pembersihan dan penggabungan data, dilakukan pemeriksaan dan penanganan nilai null yang muncul sebagai akibat dari join antar tabel.
- **Alasan:** Missing value dapat mengganggu proses representasi teks dan pemodelan TF-IDF.

#### 4. Penggabungan Genre dan Tag
- Kolom `genres` dan `tags` digabungkan ke dalam satu kolom baru bernama `text`, yang berisi deskripsi gabungan konten film.
- **Alasan:** Menyatukan semua deskripsi konten dalam satu kolom mempermudah representasi vektorisasi menggunakan TF-IDF.

#### 5. Mengatasi Duplikasi
- Dilakukan pemeriksaan terhadap duplikasi data untuk memastikan hanya satu representasi konten per film.
- **Alasan:** Menghindari bias dalam perhitungan kemiripan antar film.

#### 6. Preprocessing Teks
- Melakukan transformasi lowercasing terhadap kolom `title` dan `text` untuk standarisasi dan mengurangi redundansi kata.
- **Alasan:** Menurunkan variabilitas kata akibat perbedaan kapitalisasi dalam vektorisasi.

#### 7. TF-IDF Vectorization
- Menerapkan `TfidfVectorizer` dari scikit-learn untuk mentransformasikan kolom `text` menjadi representasi vektor numerik.
- **Alasan:** TF-IDF digunakan untuk menilai seberapa penting kata dalam dokumen tertentu relatif terhadap seluruh dokumen.

#### 8. Cosine Similarity
- Menghitung kemiripan antar film menggunakan cosine similarity pada vektor hasil TF-IDF.
- **Alasan:** Cosine similarity cocok digunakan dalam sistem rekomendasi berbasis konten untuk mengukur kesamaan antar item.

### Collaborative Filtering

#### 1. Membuat List dan Encoding Variabel userId dan movieId
- Membuat userID dan movieID menjadi list tanpa nilai yang sama.
- Menggunakan `LabelEncoder` untuk mengubah ID pengguna dan film menjadi integer unik.
- Mendapatkan beberapa informasi baru seperti jumlah user, jumlah resto, minimal rating, dan maksimal rating.
- **Alasan:** List akan digunakan pada tahap pemodelan, disamping itu encoding model pembelajaran mesin memerlukan representasi numerik.

#### 2. Normalisasi Fitur Rating
- Pada tahap ini, dilakukan normalisasi pada fitur rating.
- **Alasan:** Tahap ini penting dalam persiapan data untuk model neural network yang akan digunakan agar data memiliki nilai minimum dan maksimum yang sama.

#### 3. Pembagian Data
- Data dibagi ke dalam `x_train`, `x_val`, `y_train`, dan `y_val` menggunakan teknik split, dengan proporsi training dan validation.
- **Alasan:** Pembagian ini digunakan untuk melatih dan menguji performa model neural network.

#### 4. Penyesuaian Tipe Data
- Mengonversi `x_train` dan `x_val` ke tipe `int32` dan label `y_train` serta `y_val` ke `float32`.
- **Alasan:** TensorFlow mengharuskan input dan output memiliki tipe data numerik spesifik agar kompatibel dengan arsitektur model.

Proses data preparation ini sangat krusial dalam menjamin kualitas input yang optimal untuk model, baik dalam pendekatan berbasis konten maupun kolaboratif. Setiap tahap dilakukan dengan tujuan meningkatkan akurasi dan relevansi hasil rekomendasi.


## Modeling
Pada tahap ini, sistem rekomendasi dibangun untuk menyelesaikan permasalahan yang telah diidentifikasi sebelumnya, yaitu memberikan rekomendasi film yang relevan kepada pengguna berdasarkan preferensi mereka. Dua pendekatan algoritma yang digunakan adalah **Content Based Filtering** dan **Collaborative Filtering**, dengan masing-masing menghasilkan Top-5 movie recommendation sebagai output sistem.

### 1. Content Based Filtering

Pendekatan ini merekomendasikan film berdasarkan kemiripan konten film yang telah disukai oleh pengguna. Fitur yang digunakan untuk membandingkan antar film adalah kombinasi antara **genre** dan **tag** yang telah diproses menjadi teks gabungan. Model ini menggunakan metode **TF-IDF Vectorizer** dan **Cosine Similarity** untuk mengukur kesamaan antar film.

**Top 5 Movie Recommendation yang cocok dengan user dengan id 119 (mirip dengan film *john wick: chapter two (2017)*):**
1. *john wick (2014)*
2. *hard-boiled (lat sau san taam) (1992)*
3. *panic room (2002)*
4. *negotiator, the (1998)*
5. *butch cassidy and the sundance kid (1969)*

**Kelebihan:**
- Tidak memerlukan data pengguna lain.
- Mampu memberikan rekomendasi bahkan untuk pengguna baru yang belum memiliki histori (cold start user).

**Kekurangan:**
- Cenderung memberikan rekomendasi yang serupa dengan item yang sudah pernah dilihat, sehingga kurang mampu memperkenalkan keragaman (sering over-specialized).
- Tidak menangkap preferensi komunitas pengguna.

### 2. Collaborative Filtering

Pendekatan ini merekomendasikan film dengan mempelajari pola interaksi antar pengguna dan item (film). Model ini dibangun dengan pendekatan **user-item matrix** yang diencoding dan dilatih menggunakan pembelajaran terstruktur (matrix factorization).

**Top 5 Movie Recommendation yang cocok dengan user dengan id 474:**
1. *jezebel (1938)* 
2. *gone baby gone (2007)* 
3. *the dark knight (2008)* 
4. *the hateful eight (2015)*
5. *who killed chea vichea? (2010)*

**Kelebihan:**
- Dapat merekomendasikan item baru yang secara konten tidak mirip, namun disukai oleh pengguna dengan preferensi serupa.
- Mampu mengenali pola komunitas atau tren preferensi pengguna.

**Kekurangan:**
- Memerlukan jumlah data interaksi pengguna yang besar agar akurat.
- Mengalami kendala pada cold start (pengguna atau item baru yang belum pernah berinteraksi).


Dengan menggunakan kedua pendekatan di atas, sistem rekomendasi menjadi lebih kuat dan mampu mengatasi kekurangan masing-masing metode. Pendekatan gabungan seperti **hybrid recommendation system** dapat menjadi solusi jangka panjang yang lebih andal dan akurat.


## Evaluation

Pada bagian ini, evaluasi dilakukan untuk mengukur kinerja model yang dikembangkan dalam sistem rekomendasi. Evaluasi dilakukan menggunakan metrik yang umum dipakai dalam sistem rekomendasi berbasis rating prediksi.

### Metrik Evaluasi yang Digunakan

Model content-based filtering dievaluasi menggunakan dua metrik utama, yaitu:

- **Precision@k**: Mengukur proporsi item yang direkomendasikan (dari top-*k*) yang benar-benar relevan atau disukai oleh pengguna.
- **Recall@k**: Mengukur proporsi item relevan (yang disukai pengguna) yang berhasil direkomendasikan dari keseluruhan item relevan yang tersedia.

Model collaborative filtering dievaluasi menggunakan dua metrik utama, yaitu:

- **Loss**: Mengukur error antara rating aktual dan rating prediksi pada data training, dihitung menggunakan fungsi *BinaryCrossentropy*.
- **Root Mean Squared Error (RMSE)**: Akar kuadrat dari *Mean Squared Error*, yang digunakan untuk mengembalikan error ke satuan asli dari rating. 

### Rumus Metrik Evaluasi yang Digunakan

Fungsi Precision@k didefinisikan sebagai berikut:

$$\text{Precision@}k = \frac{|\text{Recommended}_k \cap \text{Relevant}|}{k}$$

Fungsi Recall@k didefinisikan sebagai berikut:

$$\text{Recall@}k = \frac{|\text{Recommended}_k \cap \text{Relevant}|}{|\text{Relevant}|}$$


Fungsi Binary Crossentropy didefinisikan sebagai:

$$\text{BinaryCrossentropy} = -\frac{1}{n} \sum_{i=1}^{n} \left[y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i)\right]$$

Formula RMSE adalah sebagai berikut:

$$\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }$$

Di mana:

$$\text{Recommended}_k$$ = daftar top-*k* film yang direkomendasikan

$$\text{Relevant}$$ = film-film yang disukai user (misalnya rating ≥ 4)

$$|\cdot|$$ = jumlah item dalam himpunan

$$y_i$$ = nilai aktual

$$\hat{y}_i$$ = nilai prediksi

$$n$$ = jumlah data

### Hasil Evaluasi
#### Hasil Evaluasi Content-Based Filtering
**Precision@5 = 0.20**

**Recall@5 = 0.25**

Berdasarkan hasil evaluasi, nilai **Precision@5** sebesar **0.20** menunjukkan bahwa dari 5 film yang direkomendasikan kepada pengguna, hanya 1 film yang benar-benar relevan atau disukai oleh user. Artinya, tingkat ketepatan sistem dalam memberikan rekomendasi yang sesuai preferensi pengguna masih rendah. Sementara itu, nilai **Recall@5** sebesar **0.25** mengindikasikan bahwa dari total 4 film yang disukai oleh user, hanya 1 film yang berhasil ditangkap atau tercakup dalam daftar rekomendasi. 

#### Hasil Evaluasi Collaborative Filtering
Berikut adalah hasil evaluasi model yang divisualisasikan dalam dua grafik:

##### Evaluasi Loss/Binary Crossentropy
![Image](https://github.com/user-attachments/assets/b201912f-4032-4a0f-af1f-23ed7a99ae86)

Grafik menunjukkan *loss* dan *val_loss* pada training dan validasi. Loss training menurun hingga mendekati 0.49, sementara loss validasi stabil di sekitar 0.655.
 
##### Evaluasi RMSE
![Image](https://github.com/user-attachments/assets/0cda0d02-e0c5-415d-9b71-b8b12f4993b0)

Grafik RMSE training mencapai sekitar 0.02 dan RMSE validasi turun ke kisaran 0.285. Ini menunjukkan perbaikan signifikan dalam generalisasi dan akurasi prediksi model.

### Kesimpulan
Pada Content-Based Filtering, sistem belum mampu secara optimal mencakup sebagian besar preferensi pengguna dalam rekomendasinya. Kedua metrik ini merefleksikan bahwa sistem perlu ditingkatkan, baik dari sisi pemahaman konten maupun cakupan relevansi, untuk menghasilkan rekomendasi yang lebih akurat dan menyeluruh.

Pada Collaborative Filterting, metrik RMSE sangat sesuai untuk sistem rekomendasi berbasis rating karena menggambarkan seberapa jauh prediksi model terhadap rating aktual pengguna. Nilai RMSE yang rendah menandakan bahwa model mampu memberikan prediksi yang akurat dan relevan terhadap preferensi pengguna. Hal ini mungkin dikarenakan data yang dipakai tidak banyak, mengingat data yang terbuang saat penggabungan data cukup banyak.



**---Ini adalah bagian akhir laporan---**
