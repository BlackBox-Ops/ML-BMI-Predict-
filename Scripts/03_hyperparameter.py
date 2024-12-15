import os               # import library untuk operasi sistem seperti path folder atau file
import pandas as pd     # import library untuk pengolahan dataframe 
import joblib           # import library untuk simpan hyperparameter ke format joblib 

from sklearn.model_selection import train_test_split, GridSearchCV, KFold          # library untuk setting hyperparameter model 
from sklearn.ensemble import RandomForestClassifier                                # library untuk membuat model random forest 
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc  # library untuk mengukur akurasi model 

# buat class oop dengan nama Random Forest Tuner 
class RandomForestTuner:
    def __init__(self, data_dir, feature_columns, target_columns):
        self.data_dir = data_dir
        self.feature_columns = feature_columns
        self.target_columns = target_columns

        self.train_data = None
        self.test_data = None
        self.valid_data = None
    
    # Buat fungsi untuk load datasheet train, test dan validation 
    def load_data(self):
        # Load data train, valid dan test 
        self.train_data = pd.read_csv(os.path.join(self.data_dir, 'train.csv')) # load data train.csv 
        self.valid_data = pd.read_csv(os.path.join(self.data_dir, 'valid.csv')) # Load data valid.csv 
        self.test_data  = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))  # Load data test.csv 
    
    # Buat fungsi untuk untuk memilih fitur yang dilatih 
    def select_features(self, data):
        # Load fitur x dan y dari datasheet yang akan digunakan 
        X = data[self.feature_columns]
        y = data[self.target_columns]
        return X, y 
    
    # Buat Fungsi untuk tune hyperparameter model
    def tune_hyperparameter(self, X_train, y_train):
        # perform hyperparameter tuning using grid search cv 
        param_grid = {
            'n_estimators' : [50, 100, 200],
            'max_depth' : [None, 10, 20, 30],
            'min_samples_split' : [2, 5, 10],
            'min_samples_leaf' : [1, 2, 4], 
        }

        rf = RandomForestClassifier(random_state=42)                # Load model Random Forest Classifier
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)    # Setting KFold Parameter

        # Setting GridSearchCV Parameter
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kfold, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train) # Fitting model X (Features) and y (Target)
        
        # mencari parameter terbaik untuk model algoritma random forest 
        self.best_model = grid_search.best_estimator_ 
        # tampilkan parameter terbaik 
        print(f'Best Parameters : {grid_search.best_params_}')
    
    # Buat Fungsi untuk evaluasi hasil tunning hyperparameter 
    def evaluate_model(self, X, y, dataset_name):
        # Evaluasi model datasheet dan tampilhan hasil evaluasi 
        predictions = self.best_model.predict(X)  # Prediksi model untuk variabel x (features)
        accuracy = accuracy_score(y, predictions) # Hitung akurasi dari hasil tunning hyperparameter 
        report = classification_report(y, predictions)

        print(f'{dataset_name} accuracy : {accuracy : .4f}') # tampilkan hasil dari akurasi hyperparameter 
        print(f'{dataset_name} accuracy : :\n {report}')     # tampilkan hasil dari report pelatihan 
    
    # Buat Fungsi untuk menjalankan program 
    def run(self):
        '''
        Main execution flow for loading data training, and evaluation 
        '''

        # Load data 
        print("Loading data ....")
        self.load_data()

        # Prepare datasheet 
        print("Preparing datasheet ....")
        X_train, y_train = self.select_features(self.train_data)   # Setting training data 
        X_valid, y_valid = self.select_features(self.valid_data)   # Setting validation data 
        X_test,  y_test = self.select_features(self.test_data)     # Setting testing data 

        # Perform hyperparameter tuning 
        print("Tuning Hyperparameters ...")
        self.tune_hyperparameter(X_train, y_train) # train model using best hyperparameter random forest algorithm 

        # Evaluasi model dan test validation data 
        print("Evaluating Model ...")
        self.evaluate_model(X_valid, y_valid, "Validation Data") # validasi model 
        self.evaluate_model(X_test, y_test, "Test Data") # validasi model 

# Main Execution 
if __name__ == '__main__':
    # configuration 

    DATA_DIR = 'Data'  # Direktori Datasheet 
    FEATURE_COLUMNS = ['Gender','Height','Weight'] # Pilih Fitur Gender, Height dan Weight sebagai variabel x
    TARGET_COLUMNS = 'Index' # Pilih Fitur Index sebagai variabel y 

    # initialize and run the random tuner 
    tuner = RandomForestTuner(DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMNS)
    tuner.run()
        