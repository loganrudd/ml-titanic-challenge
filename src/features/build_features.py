"""
Loads raw data, one hot encodes categorical columns,
drops unimportant columns for training, and saves processed data.
"""
import tempfile
import os
import mlflow
import argparse
import pandas as pd
import numpy as np


def build_features(raw_data_path):
    local_dir = os.path.abspath(os.path.dirname(__file__))
    raw_data_dir = os.path.join(local_dir, raw_data_path)
    print("Processing Titanic passenger data")
    training = pd.read_csv(os.path.join(raw_data_dir, 'train.csv'))
    # One hot encoding categorical columns
    embark_dummies_titanic = pd.get_dummies(training['Embarked'])
    sex_dummies_titanic = pd.get_dummies(training['Sex'])
    pclass_dummies_titanic = pd.get_dummies(training['Pclass'], prefix="Class")
    training = training.join(
        [embark_dummies_titanic, sex_dummies_titanic, pclass_dummies_titanic])
    # dropping unimportant columns for modeling
    training = \
        training.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Sex', 'Pclass', 'male'],
                      axis=1)
    processed_data_dir = os.path.join(local_dir, '../../data/processed/')
    print("Saving processed data to {}".format(processed_data_dir))
    training.to_csv(os.path.join(processed_data_dir, 'train.csv'))
    mlflow.log_artifacts(processed_data_dir, "titanic-passengers-training-data")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--raw_data_path",
                        help='Path to raw data relative to root of project directory')
    args = parser.parse_args()

    build_features(raw_data_path=args.raw_data_path)
