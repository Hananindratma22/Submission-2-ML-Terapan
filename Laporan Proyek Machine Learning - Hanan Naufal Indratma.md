# Laporan Proyek Machine Learning - Hanan Naufal Indratma

## Domain Proyek

Kanker payudara merupakan salah satu penyebab utama kematian akibat kanker pada wanita di seluruh dunia. Menurut data dari American Cancer Society, kanker payudara menduduki peringkat pertama dalam jumlah kasus di antara semua jenis tumor ganas pada wanita . Di Indonesia, prevalensi kanker payudara juga terus meningkat, menjadikannya masalah kesehatan masyarakat yang signifikan.

Deteksi dini kanker payudara sangat penting untuk meningkatkan peluang kesembuhan dan kelangsungan hidup pasien. Namun, metode konvensional seperti mammografi memiliki keterbatasan, termasuk akurasi yang bervariasi dan ketergantungan pada interpretasi manusia, yang dapat menyebabkan kesalahan diagnosis.

Kemajuan dalam bidang kecerdasan buatan, khususnya machine learning (ML) dan deep learning (DL), telah membuka peluang baru dalam diagnosis kanker payudara. Algoritma ML dapat menganalisis data medis dalam jumlah besar untuk mengidentifikasi pola yang mungkin tidak terlihat oleh manusia, sehingga meningkatkan akurasi diagnosis.

Berbagai studi telah menunjukkan efektivitas algoritma ML dalam klasifikasi kanker payudara. Misalnya, penelitian oleh Senthilkumar et al. (2024) menunjukkan bahwa metode ensemble dan deep neural network mencapai akurasi hingga 98,2% dalam klasifikasi tumor payudara . Selain itu, algoritma seperti logistic regression dan multilayer perceptron juga menunjukkan performa yang tinggi dengan akurasi mencapai 98%. [1][2]

Referensi:
[1] Senthilkumar, K.P., et al. (2024). "A Comprehensive Analysis on the Efficacy of Machine Learning-Based Algorithms for Breast Cancer Classification." Journal of Electrical Systems.

[2] Houfani, D., et al. (2020). "Breast cancer classification using machine learning techniques: a comparative study." Medical Technologies Journal.

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara mengklasifikasikan jenis tumor payudara sebagai jinak atau ganas secara akurat dan efisien menggunakan data yang tersedia?
- Apakah pemanfaatan algoritma machine learning dapat meningkatkan performa deteksi dini kanker payudara dibandingkan metode konvensional (manual diagnosis atau rule-based)?
- Model machine learning mana yang memberikan kinerja terbaik dalam hal akurasi, f1-score, precision, dan recall dalam tugas klasifikasi kanker payudara ini?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengembangkan dan membandingkan beberapa model machine learning untuk klasifikasi kanker payudara berdasarkan dataset dari Kaggle.
- Menentukan model terbaik yang mampu mengklasifikasikan tumor secara akurat sebagai jinak (benign) atau ganas (malignant).
- Meningkatkan keandalan prediksi diagnosis awal kanker payudara dengan mempertimbangkan metrik evaluasi utama: accuracy, f1-score, precision, dan recall.

### Solution statements
- Membangun model klasifikasi menggunakan empat algoritma berbeda, yakni Random Forest, Ridge Classifieer, Support Vector Machine, dan XGBoost.
- Setiap model akan dilatih dan dievaluasi pada dataset yang sama untuk memastikan perbandingan yang adil.
- Metrik evaluasi yang digunakan meliputi
    - Accuracy: mengukur proporsi prediksi yang benar.
    - Precision: proporsi prediksi positif yang benar-benar positif.
    - Recall (Sensitivity): proporsi kasus positif yang berhasil terdeteksi.
    - F1-Score: rata-rata harmonis dari precision dan recall.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah *Breast Cancer Wisconsin (Diagnostic) Dataset*, yang tersedia secara publik melalui Kaggle. Dataset ini digunakan untuk membangun model klasifikasi guna membedakan antara tumor payudara yang jinak (*benign*) dan ganas (*malignant*) berdasarkan karakteristik nukleus sel dari hasil biopsi menggunakan aspirasi jarum halus (*fine needle aspiration*). Dataset ini dapat diakses di: [Kaggle - Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data).

Dataset terdiri dari 569 observasi dan 32 kolom, yang terdiri dari:
- 1 kolom ID pasien (`id`)
- 1 kolom target/label (`diagnosis`)
- 30 kolom fitur numerik (karakteristik inti sel) berdasarkan rata-rata, standard error, dan nilai terburuk dari hasil pengukuran.

### Variabel-variabel pada Breast Cancer Dataset adalah sebagai berikut:

#### Informasi Identitas dan Target
- **id** : Nomor identifikasi pasien, tidak relevan untuk analisis.
- **diagnosis** : Label target. Berisi dua nilai:
  - `M` = malignant (ganas)
  - `B` = benign (jinak)

#### Fitur Berdasarkan Rata-rata (Mean)
- **radius_mean** : Rata-rata jarak dari pusat inti sel ke batas luar (ukuran inti).
- **texture_mean** : Rata-rata variasi dalam intensitas tingkat abu-abu (tekstur permukaan inti).
- **perimeter_mean** : Rata-rata panjang keliling inti sel.
- **area_mean** : Rata-rata luas area inti sel.
- **smoothness_mean** : Rata-rata variasi lokal dari panjang kontur (kekasaran permukaan).
- **compactness_mean** : Ukuran seberapa padat bentuk inti (dihitung dari perimeter² / area - 1.0).
- **concavity_mean** : Rata-rata tingkat lekukan pada sisi kontur inti.
- **concave points_mean** : Jumlah titik sudut yang membentuk lekukan pada kontur inti.
- **symmetry_mean** : Rata-rata tingkat simetri inti sel.
- **fractal_dimension_mean** : Rata-rata kompleksitas bentuk kontur (dimensi fraktal dari batas).

#### Fitur Berdasarkan Standard Error (SE)
- **radius_se** : Standard error dari ukuran radius.
- **texture_se** : Standard error dari tekstur.
- **perimeter_se** : Standard error dari perimeter.
- **area_se** : Standard error dari area.
- **smoothness_se** : Standard error dari kekasaran permukaan.
- **compactness_se** : Standard error dari kekompakan inti.
- **concavity_se** : Standard error dari tingkat lekukan inti.
- **concave points_se** : Standard error dari jumlah titik lekukan.
- **symmetry_se** : Standard error dari simetri inti.
- **fractal_dimension_se** : Standard error dari kompleksitas batas.

#### Fitur Berdasarkan Nilai Terburuk (Worst / Max)
- **radius_worst** : Ukuran radius terbesar dari inti yang diamati.
- **texture_worst** : Nilai tekstur terbesar dari inti.
- **perimeter_worst** : Perimeter maksimum dari inti.
- **area_worst** : Area maksimum dari inti.
- **smoothness_worst** : Kekasaran maksimum.
- **compactness_worst** : Kekompakan maksimum.
- **concavity_worst** : Lekukan maksimum.
- **concave points_worst** : Jumlah titik lekukan maksimum.
- **symmetry_worst** : Simetri maksimum dari inti sel.
- **fractal_dimension_worst** : Kompleksitas batas tertinggi.

> **Catatan:** Semua fitur numerik ini didasarkan pada hasil analisis digital citra dari sampel jaringan menggunakan metode mikroskopis. Pemilihan fitur mean, SE, dan worst mencerminkan kestabilan, variabilitas, dan ekstremitas karakteristik dari sel tumor.

### Exploratory Data Analysis (EDA)

Untuk memahami lebih lanjut struktur dan distribusi data, dilakukan beberapa teknik visualisasi dalam tahap Exploratory Data Analysis (EDA), dengan tujuan untuk:
   - Tinjau jumlah baris dan kolom dalam dataset.  
   - Tinjau jenis data di setiap kolom .
   - Cek duplikasi data dan missing value.
   - Analisis distribusi variabel numerik dengan statistik deskriptif

Visualisasi yang dilakukan:

#### Jumlah Baris dan Kolom Dataset
```python
df.shape
```
Output:
```(569, 32)``` 

Data memilikisebanyak 569 baris dan 32 kolom.

#### Jenis Data Setiap Kolom
```python
df.info()
```
Output:
| No | Column                   | Non-Null Count | Dtype   |
|----|--------------------------|----------------|---------|
| 0  | id                       | 569            | int64   |
| 1  | diagnosis                | 569            | object  |
| 2  | radius_mean              | 569            | float64 |
| 3  | texture_mean             | 569            | float64 |
| 4  | perimeter_mean           | 569            | float64 |
| 5  | area_mean                | 569            | float64 |
| 6  | smoothness_mean          | 569            | float64 |
| 7  | compactness_mean         | 569            | float64 |
| 8  | concavity_mean           | 569            | float64 |
| 9  | concave points_mean      | 569            | float64 |
| 10 | symmetry_mean            | 569            | float64 |
| 11 | fractal_dimension_mean   | 569            | float64 |
| 12 | radius_se                | 569            | float64 |
| 13 | texture_se               | 569            | float64 |
| 14 | perimeter_se             | 569            | float64 |
| 15 | area_se                  | 569            | float64 |
| 16 | smoothness_se            | 569            | float64 |
| 17 | compactness_se           | 569            | float64 |
| 18 | concavity_se             | 569            | float64 |
| 19 | concave points_se        | 569            | float64 |
| 20 | symmetry_se              | 569            | float64 |
| 21 | fractal_dimension_se     | 569            | float64 |
| 22 | radius_worst             | 569            | float64 |
| 23 | texture_worst            | 569            | float64 |
| 24 | perimeter_worst          | 569            | float64 |
| 25 | area_worst               | 569            | float64 |
| 26 | smoothness_worst         | 569            | float64 |
| 27 | compactness_worst        | 569            | float64 |
| 28 | concavity_worst          | 569            | float64 |
| 29 | concave points_worst     | 569            | float64 |
| 30 | symmetry_worst           | 569            | float64 |
| 31 | fractal_dimension_worst  | 569            | float64 |

Berdasarkan pengecekan pada tipe data, diperoleh tipe data id adalah integer, diagnosis adalah object, serta variabel lain memiliki tipe data float.

#### Cek Duplikasi Data
```python
df.duplicated().sum()
```
Output:
```np.int64(0)``` 

Tidak ditemukan duplikasi pada dataset.

#### Cek Missing Value
```python
df.isnull().sum().sum()
```
Output:
```np.int64(0)``` 

Tidak ditemukan *missing value* pada data.

#### Statistik Deskriptif
| Variabel              | count | mean        | std        | min     | 25%      | 50%      | 75%      | max        |
|-----------------------|-------|-------------|------------|---------|----------|----------|----------|------------|
| index                 | 569.0 | 30371831.43 | 125020585.61 | 8670.0  | 869218.0 | 906024.0 | 8813129.0 | 911320502.0 |
| id                    | 569.0 | 14.13       | 3.52       | 6.98    | 11.7     | 13.37    | 15.78    | 28.11      |
| radius_mean           | 569.0 | 19.29       | 4.30       | 9.71    | 16.17    | 18.84    | 21.8     | 39.28      |
| texture_mean          | 569.0 | 91.97       | 24.30      | 43.79   | 75.17    | 86.24    | 104.1    | 188.5      |
| perimeter_mean        | 569.0 | 654.89      | 351.91     | 143.5   | 420.3    | 551.1    | 782.7    | 2501.0     |
| area_mean             | 569.0 | 0.09636     | 0.01406    | 0.05263 | 0.08637  | 0.09587  | 0.1053   | 0.1634     |
| smoothness_mean       | 569.0 | 0.10434     | 0.05281    | 0.01938 | 0.06492  | 0.09263  | 0.1304   | 0.3454     |
| compactness_mean      | 569.0 | 0.08880     | 0.07972    | 0.00000 | 0.02956  | 0.06154  | 0.1307   | 0.4268     |
| concavity_mean        | 569.0 | 0.04892     | 0.03880    | 0.00000 | 0.02031  | 0.0335   | 0.074    | 0.2012     |
| concave points_mean   | 569.0 | 0.18116     | 0.02741    | 0.10600 | 0.1619   | 0.1792   | 0.1957   | 0.3040     |
| symmetry_mean         | 569.0 | 0.06280     | 0.00706    | 0.04996 | 0.0577   | 0.06154  | 0.06612  | 0.09744    |
| fractal_dimension_mean| 569.0 | 0.40517     | 0.27731    | 0.11150 | 0.2324   | 0.3242   | 0.4789   | 2.8730     |
| radius_se             | 569.0 | 1.21685     | 0.55165    | 0.36020 | 0.8339   | 1.1080   | 1.4740   | 4.8850     |
| texture_se            | 569.0 | 2.86606     | 2.02185    | 0.75700 | 1.6060   | 2.2870   | 3.3570   | 21.98      |
| perimeter_se          | 569.0 | 40.34       | 45.49      | 6.80    | 17.85    | 24.53    | 45.19    | 542.2      |
| area_se               | 569.0 | 0.00704     | 0.00300    | 0.00171 | 0.005169 | 0.00638  | 0.008146 | 0.03113    |
| smoothness_se         | 569.0 | 0.02548     | 0.01791    | 0.00225 | 0.01308  | 0.02045  | 0.03245  | 0.1354     |
| compactness_se        | 569.0 | 0.03189     | 0.03019    | 0.00000 | 0.01509  | 0.02589  | 0.04205  | 0.3960     |
| concavity_se          | 569.0 | 0.01180     | 0.00617    | 0.00000 | 0.007638 | 0.01093  | 0.01471  | 0.05279    |
| concave points_se     | 569.0 | 0.02054     | 0.00827    | 0.00788 | 0.01516  | 0.01873  | 0.02348  | 0.07895    |



## Data Preparation
Pada tahap ini, dilakukan beberapa teknik *data preparation* guna mempersiapkan dataset agar sesuai dengan kebutuhan algoritma machine learning. Langkah-langkah dilakukan secara berurutan dan konsisten sesuai dengan notebook. 

### 1. Menghapus Kolom yang Tidak Diperlukan
Kolom `id` dihapus dari dataset karena tidak memiliki nilai informasi atau hubungan dengan target (`diagnosis`). Kolom ini hanya merupakan nomor identifikasi pasien dan tidak berguna dalam proses pembelajaran model. **Alasan** dilakukan **Penghapusan kolom tidak relevan**  adalah mencegah noise dan mengurangi dimensi dataset tanpa kehilangan informasi penting.

```python
df.drop(['id'], axis=1, inplace=True)
```

### 2. Encoding Variabel Kategorik
Kolom `diagnosis` merupakan variabel target yang bersifat kategorik dengan dua nilai: `'M'` (malignant) dan `'B'` (benign). **Alasan** dilakukan **Encoding** yakni agar label target dikenali sebagai numerik oleh algoritma. Untuk dapat diproses oleh algoritma machine learning, label ini dikonversi menjadi numerik menggunakan label encoding:
- `M` → 1
- `B` → 0

```python
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
```

### 3. Normalisasi Fitur Numerik
Sebagian besar algoritma machine learning sensitif terhadap skala fitur. Oleh karena itu, dilakukan normalisasi seluruh fitur numerik menggunakan **Min-Max Scaling**, yang akan mengubah nilai fitur ke rentang [0, 1]. Tujuannya adalah agar setiap fitur memiliki kontribusi yang seimbang dalam proses pelatihan model. **Alasan** dilakukan **Normalisasi** penting untuk menyetarakan skala fitur sehingga model tidak bias terhadap fitur dengan rentang nilai besar.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
```

### 4. Membagi Dataset menjadi Data Latih dan Data Uji
Dataset dibagi menjadi dua bagian, yaitu data latih (training) dan data uji (testing), dengan rasio 80:20. Data latih digunakan untuk melatih model, sedangkan data uji digunakan untuk mengevaluasi performa model terhadap data yang belum pernah dilihat sebelumnya. **Alasan** dilakukan **Pembagian data** memungkinkan evaluasi model secara adil dan objektif, serta mencegah overfitting.

```python
# Splitting Data df
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


## Modeling
Pada tahap ini, dilakukan proses pemodelan dengan menggunakan empat algoritma machine learning untuk menyelesaikan permasalahan klasifikasi kanker payudara. Tujuan dari tahap ini adalah membandingkan performa masing-masing algoritma dengan metrik evaluasi tertentu untuk menentukan model terbaik. Berikut adalah algoritma yang digunakan:

1. **Support Vector Machine (SVM)**
2. **Random Forest Classifier**
3. **Ridge Classifier**
4. **XGBoost Classifier**

### Proses Pemodelan

Setiap model dilatih menggunakan data latih yang telah dipersiapkan sebelumnya, dengan preprocessing berupa encoding dan normalisasi (MinMaxScaler). Model kemudian dievaluasi menggunakan data uji dengan empat metrik evaluasi utama, yakni **Accuracy**, **F1 Score**, **Precision**, dan **Recall**.

### Penjelasan Hasil Pemodelan

Setiap model dilatih menggunakan data latih yang telah diproses dengan encoding dan normalisasi MinMaxScaler. Berikut adalah penjelasan untuk masing-masing model, termasuk tahapan dan parameter yang digunakan:

#### 1. Support Vector Machine (SVM)
- **Deskripsi**: Algoritma klasifikasi yang bekerja dengan menemukan hyperplane terbaik yang memisahkan kelas-kelas data.
- **Tahapan & Parameter**:
  - Model dilatih menggunakan `SVC()` dari library `scikit-learn`.
  - Kernel yang digunakan: `'rbf'` (default), yang cocok untuk data non-linear.
  - Parameter default digunakan, namun hasil sudah sangat baik bahkan tanpa tuning.
- **Kelebihan**:
  - Efektif pada ruang dimensi tinggi.
  - Memiliki performa tinggi pada dataset yang bersih dan terpisah dengan baik.
- **Kekurangan**:
  - Kurang efisien untuk dataset yang sangat besar.
  - Sensitif terhadap pemilihan kernel dan parameter regularisasi.

#### 2. Random Forest Classifier
- **Deskripsi**: Algoritma ensemble learning berbasis decision tree yang menggabungkan beberapa pohon untuk meningkatkan akurasi.
- **Tahapan & Parameter**:
  - Model dilatih menggunakan `RandomForestClassifier()` dari `scikit-learn`.
  - Parameter utama yang digunakan merupakan parameter default model.
  - Model tidak mengalami overfitting, dengan performa tinggi pada data uji.
- **Kelebihan**:
  - Kuat terhadap overfitting.
  - Dapat menangani data dengan banyak fitur dan non-linearitas.
- **Kekurangan**:
  - Model lebih sulit diinterpretasi.
  - Waktu training relatif lebih lama dibanding model linear.

#### 3. Ridge Classifier
- **Deskripsi**: Model klasifikasi linier dengan regularisasi L2 (penalty ridge regression).
- **Tahapan & Parameter**:
  - Model dilatih menggunakan `RidgeClassifier()` dari `scikit-learn`.
  - Parameter default digunakan dalam pembuatan model.
  - Model cepat dilatih dan cocok sebagai baseline.
- **Kelebihan**:
  - Sederhana dan cepat.
  - Cocok untuk baseline model.
- **Kekurangan**:
  - Tidak dapat menangkap hubungan non-linear dalam data.

#### 4. XGBoost
- **Deskripsi**: Model boosting berbasis gradient yang populer karena akurasi dan efisiensinya.
- **Tahapan & Parameter**:
  - Model dilatih menggunakan `XGBClassifier()` dari library `xgboost`.
  - Parameter utama yang digunakan:
    - `use_label_encoder=False`
    - `eval_metric='mlogloss'`
  - Performa cukup tinggi dan konsisten, namun sedikit lebih rendah dari SVM.
- **Kelebihan**:
  - Performa tinggi dalam banyak jenis data.
  - Memiliki teknik regularisasi bawaan untuk mencegah overfitting.
- **Kekurangan**:
  - Lebih kompleks dan memerlukan tuning parameter.
  - Kurang interpretatif dibanding model linear.


### Pemilihan Model Terbaik
Berikut adalah hasil evaluasi keempat model:
| Model                | Accuracy | F1 Score | Precision | Recall   |
|----------------------|----------|----------|-----------|----------|
| Support Vector Machine | 0.973684 | 0.971863 | 0.974206  | 0.969702 |
| Random Forest          | 0.964912 | 0.962302 | 0.967257  | 0.958074 |
| Ridge Classifier       | 0.956140 | 0.952638 | 0.960473  | 0.946446 |
| XGBoost                | 0.956140 | 0.953106 | 0.955357  | 0.951032 |

Berdasarkan evaluasi pada metrik-metrik yang digunakan, **Support Vector Machine** menunjukkan performa terbaik di seluruh metrik evaluasi. Dengan mempertimbangkan bahwa F1 Score dan Recall sangat penting dalam klasifikasi kanker (untuk meminimalkan false negative), maka model **SVM** dipilih sebagai model terbaik dalam proyek ini.



## Evaluation
### Metrik Evaluasi yang Digunakan
Dalam proyek ini, masalah yang dihadapi merupakan **klasifikasi biner**, sehingga metrik evaluasi yang digunakan adalah:

1. **Akurasi (Accuracy)**  
   Mengukur proporsi prediksi yang benar terhadap total data.  
   
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
 
   Cocok digunakan ketika kelas target seimbang.

2. **Precision**  
   Mengukur seberapa akurat model saat memprediksi kelas positif.  
   
   $$\text{Precision} = \frac{TP}{TP + FP}$$
   
   Cocok jika false positive lebih berdampak untuk dihindari.

3. **Recall**  
   Mengukur kemampuan model dalam menangkap seluruh kasus positif.  
   
   $$\text{Recall} = \frac{TP}{TP + FN}$$
    
   Cocok jika false negative lebih penting untuk dihindari.

4. **F1 Score**  
   Rata-rata harmonik antara precision dan recall.  
  
   $$\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
   
   Berguna ketika terdapat ketidakseimbangan antar kelas dan penting menjaga keseimbangan precision dan recall.


### Interpretasi dan Analisis

- **Support Vector Machine (SVM)** memiliki skor tertinggi untuk semua metrik: akurasi (97.36%), F1 Score (97.18%), precision (97.42%), dan recall (96.97%). Ini menunjukkan bahwa SVM memberikan performa terbaik dalam hal klasifikasi secara keseluruhan.
- **Random Forest** juga menunjukkan performa yang tinggi, namun sedikit di bawah SVM pada setiap metrik.
- **Ridge Classifier dan XGBoost** menunjukkan hasil yang mirip, namun masih kalah dari SVM dan Random Forest.

### Kesimpulan Evaluasi
Berdasarkan seluruh metrik evaluasi, **Support Vector Machine (SVM)** dipilih sebagai **model terbaik** dalam proyek ini karena mampu memberikan keseimbangan yang sangat baik antara akurasi, presisi, dan sensitivitas (recall).
- **Accuracy**: 0.973684 (tertinggi)
- **F1 Score**: 0.971863
- **Precision**: 0.974206
- **Recall**: 0.969702

Hal ini penting untuk konteks klasifikasi biner agar baik kesalahan tipe I (false positive) maupun tipe II (false negative) dapat diminimalkan secara seimbang.

**---Ini adalah bagian akhir laporan---**