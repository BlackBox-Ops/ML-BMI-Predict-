import json
from datetime import datetime


class TrainingLogger:
    @staticmethod
    def save_training_log(
        output_dir, train_file, test_file, valid_file, features, target, library_versions, 
        best_params, cv_results, performance, start_time, exception=None):
        """Save training metadata, performance, and results to a JSON file."""
        end_time = datetime.utcnow()
        training_duration = (end_time - start_time).total_seconds()

        training_log = {
            "metadata": {
                "data_files": {
                    "train": train_file,
                    "test": test_file,
                    "valid": valid_file
                },
                "features": features,
                "target": target,
                "library_versions": library_versions
            },
            "hyperparameter_tuning": {
                "best_params": best_params,
                "cv_results": {
                    "mean_test_score": list(cv_results["mean_test_score"]),
                    "params": cv_results["params"]
                }
            },
            "performance": performance,
            "logging": {
                "training_start_time": start_time.isoformat() + "Z",
                "training_end_time": end_time.isoformat() + "Z",
                "training_duration": training_duration,
                "exception": exception
            }
        }

        log_path = f"{output_dir}/training_log.json"
        with open(log_path, "w") as log_file:
            json.dump(training_log, log_file, indent=4)
        print(f"Training log saved to {log_path}")
