import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException

class DataIngestion:
    def __init__(self):
        try:
            self.data_dir = "data"
            self.fake_data_path = os.path.join(self.data_dir, "Fake.csv")
            self.true_data_path = os.path.join(self.data_dir, "True.csv")

            self.artifact_dir = "artifacts"
            os.makedirs(self.artifact_dir, exist_ok=True)

            self.output_path = os.path.join(self.artifact_dir, "raw_news_data.csv")

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("===== Data Ingestion Started =====")

            fake_df = pd.read_csv(self.fake_data_path)
            true_df = pd.read_csv(self.true_data_path)

            fake_df["label"] = 0   # Fake
            true_df["label"] = 1   # Real

            df = pd.concat([fake_df, true_df], axis=0)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            df = df[["title", "text", "label"]]

            df.to_csv(self.output_path, index=False)

            logging.info(f"Raw data saved at {self.output_path}")
            logging.info("===== Data Ingestion Completed =====")

            return self.output_path

        except Exception as e:
            logging.error("Error in data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
