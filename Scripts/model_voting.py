# Import library yang dibutuhkan
import numpy as np  # Library untuk komputasi numerik
import pandas as pd # Library untuk pengolahan data dalam bentuk dataframe
import joblib       # Library untuk memuat model dalam format joblib

# Import library untuk membagi dataset dan melakukan tuning hyperparameter
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
# Library untuk model Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Library untuk model Support Vector Machine (SVM)
from sklearn.svm import SVC
# Library untuk model XGBClassifier (dari XGBoost)
from xgboost import XGBClassifier
# Library untuk model Random Forest dan Voting Classifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# Library untuk mengukur kinerja model
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Load dataset dari file CSV
data = pd.read_csv('../Data/preprocessing.csv')

# Pisahkan fitur (X) dan label (y) dari dataset
X = data.iloc[:, :-1]  # Ambil semua kolom kecuali kolom terakhir sebagai fitur
y = data.iloc[:, -1]   # Ambil kolom terakhir sebagai target/label

# Membagi data menjadi data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
Inisialisasi model-model yang akan digunakan untuk Voting Classifier:
- Random Forest
- Naive Bayes
- XGBoost
- Support Vector Machine
"""

# Model Random Forest dengan 100 pohon keputusan
RF = RandomForestClassifier(n_estimators=100, random_state=42)

# Model Naive Bayes
NB = GaussianNB()

# Model XGBoost dengan parameter tambahan untuk menghindari peringatan
class SklearnCompatibleXGBClassifier(XGBClassifier):
    # Menambahkan dukungan __sklearn_tags__ untuk kompatibilitas
    def __sklearn_tags__(self):
        return {"estimator_type": "classifier"}

XG = SklearnCompatibleXGBClassifier(eval_metric="mlogloss", random_state=42)

# Model Support Vector Machine dengan kernel linear
SV = SVC(probability=True, kernel="linear", random_state=42)

"""
Inisialisasi Voting Classifier untuk menggabungkan semua model di atas.
Gunakan 'soft' voting untuk rata-rata probabilitas.
"""

# Voting Classifier dengan kombinasi 4 model
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RF),  # Model Random Forest
        ('nb', NB),  # Model Naive Bayes
        ('xg', XG),  # Model XGBoost
        ('sv', SV)   # Model Support Vector Machine
    ],
    voting='soft'  # Gunakan probabilitas rata-rata untuk prediksi
)

# Latih Voting Classifier menggunakan data latih
voting_clf.fit(X_train, y_train)

# Evaluasi model pada data uji (opsional, tambahkan untuk melihat hasil)
y_pred = voting_clf.predict(X_test)

# Cetak metrik evaluasi
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Simpan model VotingClassifier ke dalam file joblib

joblib_file = "voting_classifier_model.joblib"

joblib.dump(voting_clf, joblib_file)
print(f"Model disimpan ke {joblib_file}")
