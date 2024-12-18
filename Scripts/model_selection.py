import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Fungsi untuk menemukan file
def find_file(filename, search_path='.'):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Langkah pertama: Mencari file 'bmi.csv' dalam folder '../Data' atau direktori lainnya
file_path = '../Data/bmi.csv'
if not os.path.exists(file_path):
    print(f"File '{file_path}' tidak ditemukan. Mencari file di direktori lain...")
    found_path = find_file('bmi.csv', search_path='.')
    if found_path:
        print(f"File ditemukan di: {found_path}")
        file_path = found_path
    else:
        raise FileNotFoundError("File 'bmi.csv' tidak ditemukan di sistem.")

# Membaca dataset
data = pd.read_csv(file_path)
print(f"File '{file_path}' berhasil dibaca.")

# Membaca dataset
data = pd.read_csv(file_path)
print(f"File '{file_path}' berhasil dibaca.")

# Langkah kedua: Memisahkan data menjadi train (80%) dan test (20%)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Langkah ketiga: Memisahkan train menjadi train (80%) dan valid (20%)
train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Langkah keempat: Menyimpan hasil pemisahan ke file CSV
output_dir = './Data'
os.makedirs(output_dir, exist_ok=True)  # Membuat folder jika belum ada

train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
valid_data.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

print('Data telah dipisahkan dan disimpan:')
print(f"- Train: {os.path.join(output_dir, 'train.csv')}")
print(f"- Valid: {os.path.join(output_dir, 'valid.csv')}")
print(f"- Test: {os.path.join(output_dir, 'test.csv')}")