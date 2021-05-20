import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from demo.constants import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, PREPROCESSING_PIPELINE_FILE_NAME, MODEL_DIR
from demo.functions import upload_files


def get_preprocessor_pipeline(df: pd.DataFrame) -> Pipeline:
    """
    Building transformation. Returns a pipeline which involves in multiple steps.
    Args:
        df: Dataframe

    Returns: Data transformed by pipeline

    """
    numeric_features = list(NUMERICAL_COLUMNS.keys())
    numeric_features += ['HouseAge', 'HouseAgeRemodel', 'GarageAge']

    ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                        'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'GarageQual']

    categorical_features = list(set(CATEGORICAL_COLUMNS.keys()) - set(ordinal_features))

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformers = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    ordinal_transformer_1 = Pipeline(steps=[
        ('encoder',
         OrdinalEncoder(categories=[['Fa', 'TA', 'Gd', 'Ex']], handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    ordinal_transformer_2 = Pipeline(steps=[
        ('encoder',
         OrdinalEncoder(categories=[['Po', 'Fa', 'TA', 'Gd', 'Ex']], handle_unknown='use_encoded_value',
                        unknown_value=-1))
    ])

    ordinal_transformer_3 = Pipeline(steps=[
        ('encoder',
         OrdinalEncoder(categories=[['Not Available', 'Po', 'Fa', 'TA', 'Gd', 'Ex']],
                        handle_unknown='use_encoded_value',
                        unknown_value=-1))
    ])

    ordinal_transformer_4 = Pipeline(steps=[
        ('encoder',
         OrdinalEncoder(categories=[['Not Available', 'Po', 'Fa', 'TA', 'Gd', 'Ex']],
                        handle_unknown='use_encoded_value',
                        unknown_value=-1))
    ])

    ordinal_transformer_5 = Pipeline(steps=[
        ('encoder',
         OrdinalEncoder(categories=[['Not Available', 'No', 'Mn', 'Av', 'Gd']], handle_unknown='use_encoded_value',
                        unknown_value=-1))
    ])

    ordinal_transformer_6 = Pipeline(steps=[
        ('encoder', OrdinalEncoder(categories=[['Not Available', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']],
                                   handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    ordinal_transformer_7 = Pipeline(steps=[
        ('encoder', OrdinalEncoder(categories=[['Not Available', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']],
                                   handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    ordinal_transformer_8 = Pipeline(steps=[
        ('encoder',
         OrdinalEncoder(categories=[['Po', 'Fa', 'TA', 'Gd', 'Ex']], handle_unknown='use_encoded_value',
                        unknown_value=-1))
    ])

    ordinal_transformer_9 = Pipeline(steps=[
        ('encoder',
         OrdinalEncoder(categories=[['Po', 'Fa', 'TA', 'Gd', 'Ex']], handle_unknown='use_encoded_value',
                        unknown_value=-1))
    ])

    ordinal_transformer_10 = Pipeline(steps=[
        ('encoder',
         OrdinalEncoder(categories=[['Not Available', 'Po', 'Fa', 'TA', 'Gd', 'Ex']],
                        handle_unknown='use_encoded_value',
                        unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformers, categorical_features),
            ('ord1', ordinal_transformer_1, ['ExterQual']),
            ('ord2', ordinal_transformer_2, ['ExterCond']),
            ('ord3', ordinal_transformer_3, ['BsmtQual']),
            ('ord4', ordinal_transformer_4, ['BsmtCond']),
            ('ord5', ordinal_transformer_5, ['BsmtExposure']),
            ('ord6', ordinal_transformer_6, ['BsmtFinType1']),
            ('ord7', ordinal_transformer_7, ['BsmtFinType2']),
            ('ord8', ordinal_transformer_8, ['HeatingQC']),
            ('ord9', ordinal_transformer_9, ['KitchenQual']),
            ('ord10', ordinal_transformer_10, ['GarageQual'])]
    )

    joblib.dump(preprocessor.fit(df), filename=os.path.join(MODEL_DIR, PREPROCESSING_PIPELINE_FILE_NAME))
    upload_files(filename=PREPROCESSING_PIPELINE_FILE_NAME,
                 source_file_path=os.path.join(MODEL_DIR, PREPROCESSING_PIPELINE_FILE_NAME))

    return preprocessor.fit_transform(df)
