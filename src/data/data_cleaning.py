import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(train_path, test_path):
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info(f"Data loaded successfully from {train_path} and {test_path}")
        return train_data, test_data
    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        raise
    except pd.errors.EmptyDataError:
        logging.error("EmptyDataError: One of the data files is empty.")
        raise
    except pd.errors.ParserError:
        logging.error("ParserError: There was an error parsing the data file.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while loading data: {e}")
        raise

def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]
        return " ".join(text)
    except Exception as e:
        logging.error(f"Error during lemmatization: {e}")
        raise

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        text = [word for word in str(text).split() if word not in stop_words]
        return " ".join(text)
    except Exception as e:
        logging.error(f"Error during stop word removal: {e}")
        raise

def removing_numbers(text):
    try:
        text = ''.join([char for char in text if not char.isdigit()])
        return text
    except Exception as e:
        logging.error(f"Error during number removal: {e}")
        raise

def lower_case(text):
    try:
        text = text.split()
        text = [word.lower() for word in text]
        return " ".join(text)
    except Exception as e:
        logging.error(f"Error during lowercasing: {e}")
        raise

def removing_punctuations(text):
    try:
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub('\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logging.error(f"Error during punctuation removal: {e}")
        raise

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logging.error(f"Error during URL removal: {e}")
        raise

def remove_small_sentences(df):
    try:
        df['content'] = df['content'].apply(lambda x: np.nan if len(x.split()) < 3 else x)
        df.dropna(inplace=True)
        logging.info("Removed small sentences from dataframe.")
    except Exception as e:
        logging.error(f"Error during small sentence removal: {e}")
        raise

def normalize_text(df):
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logging.info("Text normalization completed.")
        return df
    except Exception as e:
        logging.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data, test_data, data_path):
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logging.info(f"Processed data saved to {data_path}")
    except OSError as e:
        logging.error(f"OSError: Failed to save data to {data_path} - {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while saving data: {e}")
        raise

def main(train_path, test_path, data_path):
    try:
        train_data, test_data = load_data(train_path, test_path)
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)
        save_data(train_data, test_data, data_path)
        logging.info("Main workflow completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred in the main workflow: {e}")
        raise

if __name__ == "__main__":
    train_path = r'C:\Users\redhu\preet1\data\raw\train.csv'
    test_path = r'C:\Users\redhu\preet1\data\raw\test.csv'
    data_path = os.path.join('data','processed')

    main(train_path, test_path, data_path)