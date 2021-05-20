import datetime
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from demo.constants import FINAL_FEATURES_LIST, MANDATORY_INPUT_FIELDS, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, \
    MODEL_DIR, PREPROCESSING_PIPELINE_FILE_NAME, MODEL_FILE_NAME
from demo.functions import get_model_and_preprocessing_pipeline


class PredictionServer:

    def __init__(self):
        self._model = None
        self._preprocessing_pipeline = None
        self._load()

    def _load(self):
        """
        Load model and preprocessed pipeline from cloud storage
        Returns:

        """
        get_model_and_preprocessing_pipeline()
        self._model = load(os.path.join(MODEL_DIR, MODEL_FILE_NAME))
        self._preprocessing_pipeline = load(os.path.join(MODEL_DIR, PREPROCESSING_PIPELINE_FILE_NAME))

    def parse_json_data(self, data: Dict) -> Pipeline:
        """
        Parse incoming JSON response to dataframe for prediction.
        Args:
            data: User provided data in dictionary form

        Returns: Pipeline transformed data.

        """

        # Creating dataframe from incoming request data
        tdf = pd.DataFrame(data, index=[0])

        # Check which information has not been provide by user
        missing_fields = list(set(list(NUMERICAL_COLUMNS.keys()) + list(CATEGORICAL_COLUMNS.keys())) - set(tdf.columns))

        # Create features which has not been provided by user
        for field in missing_fields:
            if field in NUMERICAL_COLUMNS.keys():
                tdf[field] = np.NaN
            else:
                tdf[field] = 'Not Available'

        if len(set(MANDATORY_INPUT_FIELDS).intersection(set(tdf.columns))) != len(MANDATORY_INPUT_FIELDS):
            raise ValueError(f"{', '.join(MANDATORY_INPUT_FIELDS)} should be provided")

        tdf['MasVnrType'] = tdf['MasVnrType'].replace(['None'], 'Not Available')

        # Feature engineering on numerical columns
        year_now = datetime.datetime.now().year

        tdf['HouseAge'] = year_now - int(tdf['YearBuilt'])
        tdf['HouseAgeRemodel'] = pd.Series()
        if 'YearRemodAdd' in list(tdf.columns):
            tdf['HouseAgeRemodel'] = year_now - tdf['YearRemodAdd']
        else:
            tdf['HouseAgeRemodel'] = 0

        # Setting missing Garage built year to the Year of built.
        tdf['GarageYrBlt'] = pd.Series()
        tdf['GarageYrBlt'] = tdf['GarageYrBlt'].fillna(tdf['YearBuilt'])

        tdf['GarageAge'] = pd.Series()
        tdf['GarageAge'] = year_now - int(tdf['GarageYrBlt'])

        # Filling missing values in categorical columns
        tdf[list(CATEGORICAL_COLUMNS.keys())] = tdf[list(CATEGORICAL_COLUMNS.keys())].fillna('Not Available')

        return self._preprocessing_pipeline.transform(tdf[FINAL_FEATURES_LIST])

    def predict(self, json_data: Dict) -> List:
        parsed_data = self.parse_json_data(json_data)
        return self._model.predict(parsed_data)
