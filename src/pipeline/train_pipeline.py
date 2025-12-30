import os
import sys
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.logger import logging
from src.exception import CustomException


class TrainPipeline:
    def __init__(self):
        try:
            self.data_path = os.path.join("artifacts", "news_data.csv")
            self.artifact_dir = "artifacts"

            self.vectorizer_path = os.path.join(
                self.artifact_dir, "tfidf_vectorizer.pkl"
            )
            self.model_path = os.path.join(
                self.artifact_dir, "model.pkl"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def run_training(self):
        try:
            logging.info("===== Training Pipeline Started =====")

            # Load cleaned dataset
            df = pd.read_csv(self.data_path)

            X = df["content"]
            y = df["label"]

            # Train-test split
            
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            logging.info("Train-test split completed")

           
            # TF-IDF Vectorization
            
            tfidf = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2)
            )

            X_train_tfidf = tfidf.fit_transform(X_train)
            X_test_tfidf = tfidf.transform(X_test)

            logging.info("TF-IDF vectorization completed")

            
            # Model Training (Naive Bayes)
            
            model = MultinomialNB()
            model.fit(X_train_tfidf, y_train)

            logging.info("Naive Bayes model training completed")

           
            # Evaluation
            
            y_pred = model.predict(X_test_tfidf)

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            logging.info(f"Model Accuracy: {acc}")
            logging.info(f"Confusion Matrix:\n{cm}")
            logging.info(f"Classification Report:\n{report}")

            print("\nModel Evaluation Results")
            print("------------------------")
            print("Accuracy:", acc)
            print("\nConfusion Matrix:\n", cm)
            print("\nClassification Report:\n", report)

            # Save artifacts
            os.makedirs(self.artifact_dir, exist_ok=True)
            print("Saving model to:", self.model_path)
            print("Saving vectorizer to:", self.vectorizer_path)


            joblib.dump(tfidf, self.vectorizer_path)
            joblib.dump(model, self.model_path)

            logging.info("TF-IDF vectorizer and model saved successfully")
            logging.info("===== Training Pipeline Completed =====")

        except Exception as e:
            logging.error("Error occurred in training pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_training()
