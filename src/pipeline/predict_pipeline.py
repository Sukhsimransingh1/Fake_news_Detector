import os
import sys
import joblib
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.logger import logging
from src.exception import CustomException

nltk.download("stopwords")
nltk.download("wordnet")

class PredictPipeline:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(), "artifacts")

            self.vectorizer_path = os.path.join(
                self.artifact_dir, "tfidf_vectorizer.pkl"
            )
            self.model_path = os.path.join(
                self.artifact_dir, "model.pkl"
            )

            self.stop_words = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()

        except Exception as e:
            raise CustomException(e, sys)

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z]", " ", text)

        tokens = text.split()
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]

        return " ".join(tokens)

    def predict(self, text: str):
        try:
            logging.info("Prediction started")

            # Load artifacts
            vectorizer = joblib.load(self.vectorizer_path)
            model = joblib.load(self.model_path)

            # Preprocess input text
            cleaned_text = self.clean_text(text)

            # Vectorize
            text_tfidf = vectorizer.transform([cleaned_text])

            # Predict
            prediction = model.predict(text_tfidf)[0]

            result = "Real News" if prediction == 1 else "Fake News"

            logging.info(f"Prediction completed: {result}")

            return result

        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = PredictPipeline()

    sample_text = """
    Government announces new economic reforms aimed at boosting employment
    and stabilizing inflation over the next fiscal year.
    """

    print("Prediction:", obj.predict(sample_text))
