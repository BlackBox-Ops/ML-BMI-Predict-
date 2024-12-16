import pandas as pd 
import numpy as np 

# set jumlah data yang diinginkan 
n_samples = 100

# Generate data acak
np.random.seed(42) # untuk hasil yang konsisten 

gender = np.random.randint(0, 2, size = n_samples)    # 0 atau 1 untuk gender 
height = np.random.randint(150, 200, size= n_samples) # Tinggi antara 150 - 200 cm
weight = np.random.randint(50, 100, size=n_samples)      # index (kelas) antara 1 -3
index = np.random.randint(1,4, size=n_samples)        # index (kelas) antara 1-3

# Buat DataFrame
data = pd.DataFrame({
    'Gender' : gender,
    'Height' : height,
    'Weight' : weight,
    'Index'  : index
})

# simpan ke CSV
data.to_csv('Data/new_data.csv', index=False)
print('Data baru berhasil dibuat dan disimpan ke direktori data')
