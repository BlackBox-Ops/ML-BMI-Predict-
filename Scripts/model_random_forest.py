import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, f1_score
import matplotlib.pyplot as plt

# Kelas untuk melatih model Random Forest dengan hyperparameter yang telah ditentukan
class RandomForestTrainer:
    def __init__(self, data_dir, feature_columns, target_column, hyperparameters, output_dir):
        # Inisialisasi direktori data tempat dataset berada
        self.data_dir = data_dir  # Direktori data
        # Inisialisasi kolom fitur yang digunakan untuk pelatihan
        self.feature_columns = feature_columns  # Fitur yang akan digunakan
        # Inisialisasi kolom target yang akan diprediksi
        self.target_column = target_column  # Target
        # Inisialisasi hyperparameter terbaik yang akan digunakan untuk melatih model
        self.hyperparameters = hyperparameters  # Hyperparameter terbaik
        # Inisialisasi direktori output untuk menyimpan model, hasil evaluasi, dan visualisasi
        self.output_dir = output_dir  # Direktori output untuk menyimpan model dan hasil evaluasi

        # Placeholder untuk menyimpan data
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.model = None

        # Pastikan direktori output tersedia
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """Memuat data dari folder data."""
        self.train_data = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.valid_data = pd.read_csv(os.path.join(self.data_dir, 'valid.csv'))
        self.test_data = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))

    def select_features(self, data):
        """Memilih fitur (X) dan target (y) dari dataset."""
        X = data[self.feature_columns]
        y = data[self.target_column]
        return X, y

    def train_model(self, X_train, y_train):
        """Melatih model Random Forest dengan hyperparameter yang diberikan."""
        # Inisialisasi model Random Forest dengan hyperparameter terbaik
        self.model = RandomForestClassifier(**self.hyperparameters, random_state=42)  
        # Latih model menggunakan data training
        self.model.fit(X_train, y_train)  
        print("Model training completed.")

    def evaluate_model(self, X, y, dataset_name):
        """Evaluasi model pada dataset tertentu dan cetak hasilnya."""
        # Prediksi output menggunakan model
        predictions = self.model.predict(X)  
        # Hitung akurasi model
        accuracy = accuracy_score(y, predictions)  
        # Hitung skor F1 berbobot
        f1 = f1_score(y, predictions, average='weighted')  
        # Buat laporan klasifikasi
        report = classification_report(y, predictions)  

        # Cetak hasil evaluasi
        print(f"{dataset_name} Accuracy: {accuracy:.4f}")
        print(f"{dataset_name} F1-Score: {f1:.4f}")
        print(f"{dataset_name} Classification Report:\n{report}")

        return accuracy, f1

    def plot_roc_curve(self, X, y, dataset_name):
        """Plot dan simpan ROC curve."""
        # Cek apakah model memiliki metode predict_proba untuk probabilitas
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X)[:, 1]  # Probabilitas kelas positif
            fpr, tpr, _ = roc_curve(y, y_prob, pos_label=self.model.classes_[1])
            roc_auc = auc(fpr, tpr)  # Hitung area di bawah kurva (AUC)

            # Plot ROC curve
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {dataset_name}')
            plt.legend(loc='lower right')
            plt.grid(True)
            # Simpan ROC curve ke direktori output
            roc_path = os.path.join(self.output_dir, f'roc_curve_{dataset_name}.png')
            plt.savefig(roc_path)
            plt.close()
            print(f"ROC curve saved to {roc_path}")

    def save_model(self):
        """Simpan model ke file."""
        # Simpan model ke file joblib
        model_path = os.path.join(self.output_dir, 'random_forest_model.joblib')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def run(self):
        """Alur utama: load data, latih model, evaluasi, dan simpan hasil."""
        print("Loading data...")
        self.load_data()

        # Siapkan data training, validasi, dan testing
        X_train, y_train = self.select_features(self.train_data)
        X_valid, y_valid = self.select_features(self.valid_data)
        X_test, y_test = self.select_features(self.test_data)

        # Latih model
        print("Training model...")
        self.train_model(X_train, y_train)

        # Evaluasi pada validation dan test set
        print("Evaluating model...")
        self.evaluate_model(X_valid, y_valid, "Validation")
        self.evaluate_model(X_test, y_test, "Test")

        # Plot ROC curve untuk dataset test
        self.plot_roc_curve(X_test, y_test, "Test")

        # Simpan model ke file
        self.save_model()

if __name__ == '__main__':
    DATA_DIR = 'data'  # Folder data
    FEATURE_COLUMNS = ['Gender', 'Height', 'Weight']  # Kolom fitur
    TARGET_COLUMN = 'Index'  # Kolom target
    OUTPUT_DIR = 'output'  # Folder untuk output

    # Hyperparameter terbaik (contoh)
    BEST_PARAMS = {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }

    # Inisialisasi trainer dan jalankan
    trainer = RandomForestTrainer(DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN, BEST_PARAMS, OUTPUT_DIR)
    trainer.run()
