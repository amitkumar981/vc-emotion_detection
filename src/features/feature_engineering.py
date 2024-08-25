import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import yaml
import logging

# Configure logging
logging.basicConfig(
    filename='feature_extraction.log',
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
        logging.error(f"FileNotFoundError: The parameter file at {param_path} was not found.")
        raise
    except yaml.YAMLError:
        logging.error("YAMLError: The parameter file contains invalid YAML.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while loading parameters: {e}")
        raise

def load_data(train_path, test_path):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logging.info(f"Data loaded successfully from {train_path} and {test_path}")
        return train_df, test_df
    except FileNotFoundError:
        logging.error(f"FileNotFoundError: One or both data files at {train_path} or {test_path} were not found.")
        raise
    except pd.errors.EmptyDataError:
        logging.error("EmptyDataError: One or both data files are empty.")
        raise
    except pd.errors.ParserError:
        logging.error("ParserError: There was an error parsing the data files.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while loading data: {e}")
        raise

def handle_missing_values(train_df, test_df):
    try:
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        logging.info("Missing values handled (dropped) successfully.")
        return train_df, test_df
    except Exception as e:
        logging.error(f"Unexpected error while handling missing values: {e}")
        raise

def extract_features(train_df, test_df, max_features):
    try:
        x_train = train_df['content'].values
        y_train = train_df['sentiment'].values

        x_test = test_df['content'].values
        y_test = test_df['sentiment'].values

        vectorizer = CountVectorizer(max_features=max_features)
        vectorizer.fit(x_train)

        x_train_bow = vectorizer.transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        test_df = pd.DataFrame(x_test_bow.toarray())

        train_df['label'] = y_train
        test_df['label'] = y_test

        logging.info("Feature extraction using CountVectorizer completed successfully.")
        return train_df, test_df
    except KeyError as e:
        logging.error(f"KeyError: Missing expected columns in the data - {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while extracting features: {e}")
        raise

def save_data(train_df, test_df, data_path):
    try:
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, 'train_bow.csv'), index=False)
        test_df.to_csv(os.path.join(data_path, 'test_bow.csv'), index=False)
        logging.info(f"Processed data saved to {data_path}")
    except OSError as e:
        logging.error(f"OSError: Failed to save data to {data_path} - {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while saving data: {e}")
        raise

def main(param_path, train_path, test_path, data_path):
    try:
        params = load_params(param_path)
        max_features = params['feature_engineering']['max_features']

        train_df, test_df = load_data(train_path, test_path)
        train_df, test_df = handle_missing_values(train_df, test_df)
        train_df, test_df = extract_features(train_df, test_df, max_features)
        save_data(train_df, test_df, data_path)
        logging.info("Main workflow completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred in the main workflow: {e}")
        raise

if __name__ == "__main__":
    param_path = r'C:\Users\redhu\preet1\emotion_detection\src\data\params.yaml'
    train_path = r'C:\Users\redhu\preet1\data\processed\train_processed.csv'
    test_path = r'C:\Users\redhu\preet1\data\processed\test_processed.csv'
    data_path = os.path.join('data', 'features')

    main(param_path, train_path, test_path, data_path)