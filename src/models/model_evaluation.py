import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json
import logging

# Configure logging
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

# Console handler for logging to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler for logging errors to a file
file_handler = logging.FileHandler('error.log')
file_handler.setLevel(logging.ERROR)

# Formatter for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"FileNotFoundError: The model file at {model_path} was not found.")
        raise
    except pickle.UnpicklingError:
        logger.error(f"UnpicklingError: Failed to unpickle the model from {model_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading the model: {e}")
        raise

def load_test_data(test_path):
    try:
        test_data = pd.read_csv(test_path)
        x_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        logger.info(f"Test data loaded successfully from {test_path}")
        return x_test, y_test
    except FileNotFoundError:
        logger.error(f"FileNotFoundError: The test data file at {test_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.error("EmptyDataError: The test data file is empty.")
        raise
    except pd.errors.ParserError:
        logger.error("ParserError: There was an error parsing the test data file.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading test data: {e}")
        raise

def evaluate_model(model, x_test, y_test):
    try:
        y_pred = model.predict(x_test)
        y_test_prob = model.predict_proba(x_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_test_prob)  # Corrected: AUC requires probabilities
        }
        logger.info("Model evaluation completed successfully.")
        return metrics
    except ValueError as e:
        logger.error(f"ValueError during model evaluation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model evaluation: {e}")
        raise

def save_metrics(metrics, metrics_path):
    try:
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info(f"Metrics saved successfully to {metrics_path}")
    except OSError as e:
        logger.error(f"OSError: Failed to save the metrics to {metrics_path} - {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving metrics: {e}")
        raise

def main(model_path, test_path, metrics_path):
    try:
        model = load_model(model_path)
        x_test, y_test = load_test_data(test_path)
        metrics = evaluate_model(model, x_test, y_test)
        save_metrics(metrics, metrics_path)
        logger.info("Main workflow completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main workflow: {e}")
        raise

if __name__ == "__main__":
    model_path = r'models/model.pkl'
    test_path = r'C:\Users\redhu\preet1\data\features\test_bow.csv'
    metrics_path = 'json.metrics'  # Corrected file name

    main(model_path, test_path, metrics_path)
