# Text-Prediction-Model

## Deskripsi Proyek

Proyek ini bertujuan untuk membangun model pembelajaran mesin yang mampu memprediksi kata berikutnya dalam urutan teks menggunakan soneta Shakespeare. Dengan memanfaatkan teknik pemrosesan bahasa alami (NLP), model ini dilatih untuk mengenali pola dalam teks dan menghasilkan prediksi kata yang akurat.

## Dataset

Dataset yang digunakan dalam proyek ini adalah kumpulan soneta Shakespeare, yang terdiri dari lebih dari 2000 baris teks. Dataset ini menyediakan beragam contoh gaya bahasa Shakespeare yang kaya, yang sangat berguna untuk melatih model prediksi teks.[LINK DATASET](https://www.google.com/url?q=https%3A%2F%2Fwww.opensourceshakespeare.org%2Fviews%2Fsonnets%2Fsonnet_view.php%3Frange%3Dviewrange%26sonnetrange1%3D1%26sonnetrange2%3D154)

## Langkah-langkah Proyek

1. **Persiapan Data:**
   - Memuat dan membersihkan dataset.
   - Membuat urutan n-gram dari teks.
   - Menambahkan padding pada urutan n-gram untuk mencapai panjang maksimum yang diinginkan.

2. **Pemisahan Data Menjadi Fitur dan Label:**
   - Fitur adalah urutan n-gram yang telah dipad dengan kata terakhir dihapus.
   - Label adalah kata terakhir yang dihapus dari urutan n-gram, yang dienkoding dalam bentuk one-hot.

3. **Membuat Model:**
   - Menggunakan arsitektur model yang terdiri dari:
     - Layer Embedding dengan output_dim=100.
     - Layer GRU Bidirectional.
     - Layer Dense dengan jumlah unit yang sama dengan total jumlah kata dalam korpus dan menggunakan fungsi aktivasi softmax.

4. **Pelatihan Model:**
   - Model dilatih menggunakan fitur dan label yang telah diproses sebelumnya selama 50 epoch.

5. **Pengujian Model:**
   - Menghasilkan teks baru berdasarkan teks seed yang diberikan.
   - Model memprediksi kata berikutnya dan menambahkan ke teks seed secara iteratif.

## Hasil

Model berhasil mencapai akurasi sebesar 84.54% setelah 50 epoch pelatihan. Model dapat menghasilkan teks baru yang mengikuti gaya bahasa Shakespeare, meskipun masih terdapat beberapa ketidakkoherenan dan pengulangan dalam prediksi kata.

## Contoh Penggunaan

Untuk menguji model, gunakan teks seed dan hasilkan prediksi kata berikutnya:

```python
import numpy as np

# Teks seed untuk memulai prediksi
seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100

for _ in range(next_words):
    # Konversi teks menjadi urutan
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    # Pad urutan
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    # Dapatkan probabilitas untuk memprediksi kata
    predicted = model.predict(token_list, verbose=0)
    # Pilih kata berikutnya berdasarkan probabilitas maksimum
    predicted = np.argmax(predicted, axis=-1).item()
    # Dapatkan kata aktual dari indeks kata
    output_word = tokenizer.index_word[predicted]
    # Tambahkan ke teks saat ini
    seed_text += " " + output_word

print(seed_text)
```

## Skill yang Digunakan

- **Pemrosesan Bahasa Alami (NLP)**
- **Pembelajaran Mesin (Machine Learning)**
- **Pemrograman Python**
- **Penggunaan Keras dan TensorFlow**

## Penjelasan Lebih Lanjut

Untuk penjelasan lebih lanjut tentang proyek ini, kunjungi artikel Medium berikut: [Membangun Model Prediksi Kata Menggunakan Machine Learning](https://medium.com/@silviadharma07/membangun-model-prediksi-kata-menggunakan-machine-learning-5281b30f7ca6)

