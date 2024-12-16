import joblib
import pandas as pd

class RandomForestDirectTester:
    def __init__(self, model_path, input_data):
        """
        :param model_path: Path to the saved Random Forest model (joblib format).
        :param input_data: List of dictionaries containing input data.
        """
        self.model_path = model_path
        self.input_data = input_data
        self.model = None

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

        # Display predictions
        print("Predictions:")
        for i, prediction in enumerate(predictions):
            print(f"Data {i+1}: Predicted Class = {prediction}")

    def run(self):
        """Main execution flow for testing."""
        print("Loading model...")
        self.load_model()

        print("Making predictions...")
        self.predict()

if __name__ == "__main__":
    # Path to the saved model
    MODEL_PATH = "output/random_forest_model.joblib"

    # Input data for testing
    # Replace the values below with your own data (must match model's feature set)
    INPUT_DATA = [
        {"Gender": 1, "Height": 175, "Weight": 70},
        {"Gender": 0, "Height": 160, "Weight": 55},
        {"Gender": 1, "Height": 180, "Weight": 80},
        {"Gender": 0, "Height": 150, "Weight": 50},
        {"Gender": 1, "Height": 165, "Weight": 65}
    ]

    # Initialize and run the tester
    tester = RandomForestDirectTester(MODEL_PATH, INPUT_DATA)
    tester.run()
