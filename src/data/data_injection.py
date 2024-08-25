import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def load_params(param_path):
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except FileNotFoundError:
        print(f"Error: The parameter file at {param_path} was not found.")
        raise
    except yaml.YAMLError:
        print("Error: The parameter file contains invalid YAML.")
        raise
    except Exception as e:
        print(f"Unexpected error while loading parameters: {e}")
        raise

def load_data(data_url):
    try:
        df = pd.read_csv(data_url)
        return df
    except pd.errors.EmptyDataError:
        print("Error: The data file is empty.")
        raise
    except pd.errors.ParserError:
        print("Error: There was an error parsing the data file.")
        raise
    except Exception as e:
        print(f"Unexpected error while loading data from {data_url}: {e}")
        raise

def preprocess_data(df):
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        return final_df
    except KeyError as e:
        print(f"Error: Missing expected columns in the data - {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during preprocessing: {e}")
        raise

def split_data(final_df, test_size, random_state=42):
    try:
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=random_state)
        return train_data, test_data
    except ValueError as e:
        print(f"Error during data splitting: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during data splitting: {e}")
        raise

def save_data(train_data, test_data, data_path):
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
    except OSError as e:
        print(f"Error: Failed to save data to {data_path} - {e}")
        raise
    except Exception as e:
        print(f"Unexpected error while saving data: {e}")
        raise

def main(param_path, data_url, data_dir):
    try:
        params = load_params(param_path)
        test_size = params['data_injection']['test_size']

        df = load_data(data_url)
        final_df = preprocess_data(df)
        train_data, test_data = split_data(final_df, test_size)

        save_data(train_data, test_data, data_dir)
    except Exception as e:
        print(f"An error occurred in the main workflow: {e}")
        raise

if __name__ == "__main__":
    param_path = r'C:\Users\redhu\preet1\emotion_detection\src\data\params.yaml'
    data_url = r'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
    data_dir = os.path.join('data','raw')

    main(param_path, data_url, data_dir)