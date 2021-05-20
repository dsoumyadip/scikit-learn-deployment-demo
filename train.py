# Import dependencies
import datetime
import os

import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from demo.constants import DATA_DIR, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, MODEL_DIR, FINAL_FEATURES_LIST, \
    TRAIN_FILE_NAME, MODEL_FILE_NAME
from demo.functions import get_train_data, upload_files
from demo.logger import logger
from demo.preprocessor import get_preprocessor_pipeline


def train():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    logger.info("Downloading data from cloud storage...")
    get_train_data()

    logger.info(f"Reading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE_NAME))

    logger.info(f"There are total {df.shape[0]} Rows and {df.shape[1]} columns")

    # Drop Id column
    df.drop(columns=['Id'], inplace=True)

    # Setting missing Garage built year to the Year of built.
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

    # Feature engineering on numerical columns
    year_now = datetime.datetime.now().year

    df['HouseAge'] = year_now - df['YearBuilt']
    df['HouseAgeRemodel'] = year_now - df['YearRemodAdd']
    df['GarageAge'] = year_now - df['GarageYrBlt']

    # Dropping some columns which is not required
    df.drop(columns=['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], inplace=True)

    # In some categorical column there are lot of missing values. we'll remove them
    df.drop(columns=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)

    # Replacing some value for better data representation
    df['MasVnrType'] = df['MasVnrType'].replace(['None'], 'Not Available')

    # Filling missing values in categorical columns
    df[list(CATEGORICAL_COLUMNS.keys())] = df[list(CATEGORICAL_COLUMNS.keys())].fillna('Not Available')

    # Change data type of numerical columns
    df = df.astype(NUMERICAL_COLUMNS)
    df = df.astype(CATEGORICAL_COLUMNS)

    logger.info(f"Preprocessing started...")
    preprocessed_data = get_preprocessor_pipeline(df[FINAL_FEATURES_LIST])
    logger.info(f"Preprocessing has been completed.")

    model = Pipeline(steps=[('regressor', LinearRegression()),
                            ])

    # Splitting train and test data
    x_train, x_test, y_train, y_test = train_test_split(preprocessed_data, df['SalePrice'],
                                                        test_size=0.2, shuffle=True)

    logger.info("Training the model...")
    model.fit(x_train, y_train)
    logger.info("Training has been completed.")

    logger.info("Evaluating performance of model...")
    score_result = model.score(x_test, y_test)
    logger.info(score_result)

    full_path = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
    dump(model, full_path)
    upload_files(full_path, MODEL_FILE_NAME)


if __name__ == '__main__':
    train()
