import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

class RandomForestDirectTester:
    def __init__(self, model_path, input_data, class_mapping):
        """
        :param model_path: Path to the saved Random Forest model (joblib format).
        :param input_data: List of dictionaries containing input data.
        :param class_mapping: Dictionary mapping class integers to their descriptions.
        """
        self.model_path = model_path
        self.input_data = input_data
        self.class_mapping = class_mapping
        self.model = None
        self.true_labels = [0, 1, 2, 3, 4, 5]  # Example true labels for evaluation

    def load_model(self):
        """Load the Random Forest model from the joblib file."""
        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def predict(self):
        """Make predictions on the input data and display the results."""
        # Convert input data to a DataFrame
        input_df = pd.DataFrame(self.input_data)
        print("Input Data:")
        print(input_df)

        # Make predictions
        predictions = self.model.predict(input_df)

        # Display predictions with class descriptions
        
        print("Predictions:")
        for i, prediction in enumerate(predictions):
            class_description = self.class_mapping.get(prediction, "Unknown")
            print(f"Data {i+1}: Predicted Class = {prediction} ({class_description})")

        return predictions

    def evaluate(self, predictions):
        """Evaluate the model using accuracy and F1 score."""
        # Calculate accuracy and F1 score
        # accuracy = accuracy_score(self.true_labels, predictions)
        # f1 = f1_score(self.true_labels, predictions, average='weighted')

        # print(f"Accuracy: {accuracy:.4f}")
        # print(f"F1 Score: {f1:.4f}")

    def run(self):
        """Main execution flow for testing."""
        print("Loading model...")
        self.load_model()

        # print("Making predictions...")
        predictions = self.predict()

        # print("Evaluating model...")
        # self.evaluate(predictions)

if __name__ == "__main__":
    # Path to the saved model
    MODEL_PATH = "output/random_forest_model.joblib"

    # Input data for testing
    # Replace the values below with your own data (must match model's feature set)
    INPUT_DATA = [
        {"Gender": 1, "Height": 175, "Weight": 70},
        {"Gender": 0, "Height": 160, "Weight": 55},
        {"Gender": 1, "Height": 159, "Weight": 70},
        {"Gender": 0, "Height": 150, "Weight": 50},
        {"Gender": 1, "Height": 173, "Weight": 131}
    ]

    # Class mapping
    CLASS_MAPPING = {
        0: "Extremely Weak",
        1: "Weak",
        2: "Normal",
        3: "Overweight",
        4: "Obesity",
        5: "Extremely Obese"
    }

    # Initialize and run the tester
    tester = RandomForestDirectTester(MODEL_PATH, INPUT_DATA, CLASS_MAPPING)
    tester.run()
