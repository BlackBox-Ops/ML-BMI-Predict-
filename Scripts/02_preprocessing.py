# Import library yang akan digunakan 
import os           # Library untuk operasi sistem
import pandas as pd # Library untuk pengolahan dataframe
from sklearn.preprocessing import LabelEncoder  # Library untuk preprocessing label kategorik ke numerik

# Fungsi untuk menemukan file 
def find_file(filename, search_path='.'):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Langkah pertama : mencari file bmi.csv dalam folder ../Data
file_path = '../Data/bmi.csv'
if not os.path.exists(file_path):
    print(f"File '{file_path}' tidak ditemukan. Mencari file di direktori lain...")
    found_path = find_file('bmi.csv', search_path='.')
    if found_path:
        print(f"File ditemukan di: {found_path}")
        file_path = found_path
    else:
        raise FileNotFoundError("File 'bmi.csv' tidak ditemukan di sistem.")

# Membaca datasheet
data = pd.read_csv(file_path)
print(f"File '{file_path}' berhasil dibaca.")

'''
Kita akan merubah kolom Gender dari yang tadinya 
memiliki tipe data object menjadi tipe data numerik 
agar fitur Gender bisa digunakan untuk membuat model.
'''

# Definisikan variabel untuk label encoder 
labelencoder = LabelEncoder()

# Preprocessing kolom Gender 
data['Gender'] = labelencoder.fit_transform(data['Gender'])

# Direktori output
output_dir = '../Data'

# Membuat direktori output jika belum ada
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {os.path.abspath(output_dir)}")

# Menyimpan data ke file CSV setelah preprocessing
output_path = os.path.join(output_dir, 'preprocessing.csv')  # Gabungkan path
data.to_csv(output_path, index=False)  # Simpan file CSV tanpa index

# Validasi hasil penyimpanan
if os.path.exists(output_path):
    print(f"File berhasil disimpan di: {os.path.abspath(output_path)}")
else:
    print(f"File tidak ditemukan di: {os.path.abspath(output_path)}")

# Melihat hasil preprocessing
print(data.head())
print(f"File preprocessing telah disimpan di: {output_path}")
