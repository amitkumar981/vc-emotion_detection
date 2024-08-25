import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging
import os

# Configure logging
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_params(param_path):
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded successfully from {param_path}")
        return params
    except FileNotFoundError:
        logging.error(f"The parameter file at {param_path} was not found.")
        raise
    except yaml.YAMLError:
        logging.error(f"The parameter file at {param_path} contains invalid YAML.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while loading parameters: {e}")
        raise

def load_data(train_path):
    try:
        df = pd.read_csv(train_path)
        logging.info(f"Training data loaded successfully from {train_path}")
        return df
    except FileNotFoundError:
        logging.error(f"The training data file at {train_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"The training data file at {train_path} is empty.")
        raise
    except pd.errors.ParserError:
        logging.error(f"There was an error parsing the training data file at {train_path}.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while loading training data: {e}")
        raise

def extract_features_and_labels(df):
    try:
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        logging.info("Features and labels extracted successfully from the dataframe")
        return x, y
    except IndexError as e:
        logging.error(f"Dataframe does not have the expected structure: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during feature and label extraction: {e}")
        raise

def train_model(x_train, y_train, n_estimators):
    try:
        clf = GradientBoostingClassifier(n_estimators=n_estimators)
        clf.fit(x_train, y_train)
        logging.info(f"Model trained successfully with n_estimators={n_estimators}")
        return clf
    except ValueError as e:
        logging.error(f"Error during model training: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during model training: {e}")
        raise

def save_model(model, model_path):
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f"Model saved successfully at {model_path}")
    except OSError as e:
        logging.error(f"Failed to save the model to {model_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while saving the model: {e}")
        raise

def main(param_path, train_path, model_path):
    try:
        logging.info("Starting main workflow")
        params = load_params(param_path)
        n_estimators = params.get('model_building', {}).get('n_estimators', 50)
        logging.info(f"Using n_estimators={n_estimators} for model training")

        train_df = load_data(train_path)
        x_train, y_train = extract_features_and_labels(train_df)

        model = train_model(x_train, y_train, n_estimators)
        save_model(model, model_path)
        logging.info("Main workflow completed successfully")
    except Exception as e:
        logging.error(f"An error occurred in the main workflow: {e}")
        raise

if __name__ == "__main__":
    param_path = r'C:\Users\redhu\preet1\emotion_detection\src\data\params.yaml'
    train_path = r'data/features/train_bow.csv'
    model_path = r'models/model.pkl'

    main(param_path, train_path, model_path)
