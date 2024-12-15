import os               # Library untuk operasi sistem seperti path folder atau file
import pandas as pd     # Library untuk pengolahan data dalam bentuk DataFrame
import joblib           # Library untuk menyimpan model dengan format joblib
import matplotlib.pyplot as plt  # Library untuk visualisasi

from datetime import datetime  # Library untuk mengelola waktu
from sklearn.model_selection import train_test_split, GridSearchCV, KFold  # Library untuk split data dan tuning model
from sklearn.ensemble import RandomForestClassifier  # Library untuk algoritma Random Forest
from sklearn.metrics import (
    accuracy_score, classification_report, roc_curve, auc, f1_score
)  # Library untuk evaluasi model
from log import TrainingLogger  # Modul untuk log hasil training

class RandomForestTuner:
    def __init__(self, data_dir, feature_columns, target_column, output_dir):
        # Inisialisasi atribut class
        self.data_dir = data_dir                # Direktori data
        self.feature_columns = feature_columns  # Kolom fitur (X)
        self.target_column = target_column      # Kolom target (y)
        self.output_dir = output_dir            # Direktori output untuk menyimpan hasil

        self.train_data = None               # Placeholder untuk data training
        self.test_data = None                # Placeholder untuk data testing
        self.valid_data = None               # Placeholder untuk data validasi
        self.best_model = None               # Placeholder untuk model terbaik
        self.start_time = datetime.utcnow()  # Waktu mulai training

        # Pastikan direktori output tersedia
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """Load train, validation, dan test datasets."""
        self.train_data = pd.read_csv(os.path.join(self.data_dir, "train.csv"))  # Load data training
        self.valid_data = pd.read_csv(os.path.join(self.data_dir, "valid.csv"))  # Load data validasi
        self.test_data = pd.read_csv(os.path.join(self.data_dir, "test.csv"))    # Load data testing

    def select_features(self, data):
        """Pilih fitur (X) dan target (y) dari dataset."""
        X = data[self.feature_columns]  # Pilih kolom fitur
        y = data[self.target_column]    # Pilih kolom target
        return X, y

    def tune_hyperparameter(self, X_train, y_train):
        """Melakukan hyperparameter tuning menggunakan GridSearchCV."""
        param_grid = {
            "n_estimators": [50, 100, 200],   # Jumlah pohon dalam Random Forest
            "max_depth": [None, 10, 20, 30],  # Kedalaman maksimum pohon
            "min_samples_split": [2, 5, 10],  # Minimum sampel untuk split
            "min_samples_leaf": [1, 2, 4],    # Minimum sampel per daun
        }

        rf = RandomForestClassifier(random_state=42)              # Inisialisasi model Random Forest
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Inisialisasi KFold Cross Validation

        grid_search = GridSearchCV(
            estimator=rf,                  # Model yang akan di-tuning
            param_grid=param_grid,         # Grid parameter
            cv=kfold,                      # Cross-validation
            scoring="accuracy",            # Metode evaluasi
            verbose=1,                     # Tingkat logging
            n_jobs=-1,                     # Gunakan semua prosesor yang tersedia
        )
        grid_search.fit(X_train, y_train)  # Fit model dengan data training

        self.best_model = grid_search.best_estimator_   # Simpan model terbaik
        self.cv_results = grid_search.cv_results_       # Simpan hasil cross-validation
        self.best_params = grid_search.best_params_     # Simpan parameter terbaik
        print(f"Best Parameters: {grid_search.best_params_}")

    def evaluate_model(self, X, y, dataset_name):
        """Evaluasi model pada dataset tertentu dan cetak hasilnya."""
        predictions = self.best_model.predict(X)           # Prediksi data
        accuracy = accuracy_score(y, predictions)          # Hitung akurasi
        f1 = f1_score(y, predictions, average="weighted")  # Hitung F1-Score
        report = classification_report(y, predictions)     # Laporan klasifikasi

        print(f"{dataset_name} Accuracy: {accuracy:.4f}")          # Cetak akurasi
        print(f"{dataset_name} F1-Score: {f1:.4f}")                # Cetak F1-Score
        print(f"{dataset_name} Classification Report:\n{report}")  # Cetak laporan klasifikasi

        return accuracy, f1

    def plot_roc_curve(self, X, y, dataset_name):
        """Plot dan simpan ROC curve untuk dataset tertentu."""
        if hasattr(self.best_model, "predict_proba"):
            y_prob = self.best_model.predict_proba(X)[:, 1]  # Probabilitas prediksi
            fpr, tpr, _ = roc_curve(y, y_prob, pos_label=self.best_model.classes_[1])  # Hitung ROC
            roc_auc = auc(fpr, tpr)  # Hitung AUC

            plt.figure()
            plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")  # Plot ROC
            plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Garis diagonal
            plt.xlabel("False Positive Rate")            # Label sumbu X
            plt.ylabel("True Positive Rate")             # Label sumbu Y
            plt.title(f"ROC Curve - {dataset_name}")     # Judul plot
            plt.legend(loc="lower right")                # Lokasi legenda
            plt.grid(True)                               # Tambahkan grid
            plt.savefig(os.path.join(self.output_dir, f"roc_curve_{dataset_name}.png"))  # Simpan plot
            plt.close()

    def plot_training_results(self):
        """Plot dan simpan hasil training (mean test scores)."""
        mean_test_scores = self.cv_results["mean_test_score"]  # Rata-rata skor tes
        params = range(len(mean_test_scores))                  # Indeks parameter

        plt.figure()
        plt.plot(params, mean_test_scores, marker="o", label="Mean Test Accuracy")  # Plot skor
        plt.xlabel("Parameter Set")       # Label sumbu X
        plt.ylabel("Accuracy")            # Label sumbu Y
        plt.title("Grid Search Results")  # Judul plot
        plt.legend()    # Tambahkan legenda
        plt.grid(True)  # Tambahkan grid
        plt.savefig(os.path.join(self.output_dir, "training_results.png"))  # Simpan plot
        plt.close()

    def save_model(self):
        """Simpan model terbaik ke file."""
        model_path = os.path.join(self.output_dir, "best_random_forest_model.joblib")  # Path file model
        joblib.dump(self.best_model, model_path)  # Simpan model
        print(f"Model saved to {model_path}")     # Cetak informasi penyimpanan

    def run(self):
        """Alur utama untuk memuat data, melatih, dan mengevaluasi model."""
        try:
            self.load_data()  # Load data

            # Siapkan dataset
            X_train, y_train = self.select_features(self.train_data)  # Data training
            X_valid, y_valid = self.select_features(self.valid_data)  # Data validasi
            X_test, y_test = self.select_features(self.test_data)     # Data testing

            # Tuning hyperparameter
            self.tune_hyperparameter(X_train, y_train)  # Latih model

            # Evaluasi model
            valid_accuracy, valid_f1 = self.evaluate_model(X_valid, y_valid, "Validation")
            test_accuracy, test_f1 = self.evaluate_model(X_test, y_test, "Test")

            # Simpan visualisasi
            self.plot_roc_curve(X_test, y_test, "Test")  # Plot ROC
            self.plot_training_results()  # Plot hasil training

            # Simpan model
            self.save_model()  # Simpan model terbaik

            # Simpan log training
            TrainingLogger.save_training_log(
                output_dir=self.output_dir,  # Direktori output
                train_file=os.path.join(self.data_dir, "train.csv"),
                test_file=os.path.join(self.data_dir, "test.csv"),
                valid_file=os.path.join(self.data_dir, "valid.csv"),
                features=self.feature_columns,  # Kolom fitur
                target=self.target_column,      # Kolom target
                library_versions={
                    "pandas": pd.__version__,
                    "joblib": joblib.__version__,
                    "sklearn": "1.3.1",
                    "matplotlib": plt.matplotlib.__version__,
                },
                best_params=self.best_params,           # Parameter terbaik
                cv_results=self.cv_results,             # Hasil cross-validation
                performance={
                    "validation_accuracy": valid_accuracy,
                    "validation_f1_score": valid_f1,
                    "test_accuracy": test_accuracy,
                    "test_f1_score": test_f1,
                },
                start_time=self.start_time,  # Waktu mulai
            )
        except Exception as e:
            # Simpan log jika terjadi kesalahan
            TrainingLogger.save_training_log(
                output_dir=self.output_dir,
                train_file=os.path.join(self.data_dir, "train.csv"),
                test_file=os.path.join(self.data_dir, "test.csv"),
                valid_file=os.path.join(self.data_dir, "valid.csv"),
                features=self.feature_columns,
                target=self.target_column,
                library_versions={
                    "pandas": pd.__version__,
                    "joblib": joblib.__version__,
                    "sklearn": "1.3.1",
                    "matplotlib": plt.matplotlib.__version__,
                },
                best_params=None,               # Parameter terbaik tidak ada karena error
                cv_results=None,                # Hasil cross-validation kosong
                performance={},                 # Kinerja kosong
                start_time=self.start_time,
                exception=str(e),               # Pesan kesalahan
            )
            raise  # Lempar ulang error

if __name__ == "__main__":
    # Konfigurasi awal
    DATA_DIR = "Data"  # Direktori data
    FEATURE_COLUMNS = ["Gender", "Height", "Weight"]  # Kolom fitur
    TARGET_COLUMN = "Index"  # Kolom target
    OUTPUT_DIR = "logs"      # Direktori logs

    # Inisialisasi tuner dan jalankan
    tuner = RandomForestTuner(DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN, OUTPUT_DIR)
    tuner.run()