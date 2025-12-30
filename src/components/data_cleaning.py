import os
import sys
import re
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.logger import logging
from src.exception import CustomException

nltk.download("stopwords")
nltk.download("wordnet")


class DataCleaning:
    def __init__(self):
        try:
            self.input_path = os.path.join("artifacts", "raw_news_data.csv")
            self.output_path = os.path.join("artifacts", "news_data.csv")

        except Exception as e:
            raise CustomException(e, sys)

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z]", " ", text)

        tokens = text.split()
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stop_words
        ]

        return " ".join(tokens)

    def initiate_data_cleaning(self):
        try:
            logging.info("===== Data Cleaning Started =====")

            df = pd.read_csv(self.input_path)

            df["content"] = df["title"] + " " + df["text"]
            df["content"] = df["content"].apply(self.clean_text)

            df = df[["content", "label"]]
            df.to_csv(self.output_path, index=False)
            logging.info(f"Cleaned data saved at {self.output_path}")
            logging.info("===== Data Cleaning Completed =====")

            return self.output_path

        except Exception as e:
            logging.error("Error in data cleaning")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataCleaning()
    obj.initiate_data_cleaning()
