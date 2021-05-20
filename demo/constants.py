import os

BUCKET_NAME = 'scikit-learn-demo'
TRAIN_FILE_NAME = 'train.csv'

DATA_DIR = os.path.join(os.getcwd(), 'data')
LOG_DIR = os.path.join(os.getcwd(), 'logs')
MODEL_DIR = os.path.join(os.getcwd(), 'model')

MODEL_FILE_NAME = 'scikit-learn-demo.joblib'
PREPROCESSING_PIPELINE_FILE_NAME = 'scikit-learn-demo-preprocessing-pipeline.joblib'

# Having a first look at the data we have decided numerical and categorical columns
NUMERICAL_COLUMNS = {'LotFrontage': 'float',
                     'LotArea': 'float',
                     'OverallQual': 'int',
                     'OverallCond': 'int',
                     # 'YearBuilt': 'int',
                     # 'YearRemodAdd': 'int',
                     'MasVnrArea': 'float',
                     'BsmtFinSF1': 'float',
                     'BsmtFinSF2': 'float',
                     'BsmtUnfSF': 'float',
                     'TotalBsmtSF': 'float',
                     '1stFlrSF': 'float',
                     '2ndFlrSF': 'float',
                     'LowQualFinSF': 'float',
                     'GrLivArea': 'float',
                     'BsmtFullBath': 'int',
                     'BsmtHalfBath': 'int',
                     'FullBath': 'int',
                     'HalfBath': 'int',
                     'BedroomAbvGr': 'int',
                     'KitchenAbvGr': 'int',
                     'TotRmsAbvGrd': 'int',
                     'Fireplaces': 'int',
                     # 'GarageYrBlt': 'int',
                     'GarageCars': 'int',
                     'GarageArea': 'float',
                     'WoodDeckSF': 'float',
                     'OpenPorchSF': 'float',
                     'EnclosedPorch': 'float',
                     '3SsnPorch': 'float',
                     'ScreenPorch': 'float',
                     'PoolArea': 'float',
                     'MiscVal': 'float',
                     }

CATEGORICAL_COLUMNS = {'MSSubClass': 'string',
                       'MSZoning': 'string',
                       'Street': 'string',
                       # 'Alley': 'string',
                       'LotShape': 'string',
                       'LandContour': 'string',
                       'Utilities': 'string',
                       'LotConfig': 'string',
                       'LandSlope': 'string',
                       'Neighborhood': 'string',
                       'Condition1': 'string',
                       'Condition2': 'string',
                       'BldgType': 'string',
                       'HouseStyle': 'string',
                       'RoofStyle': 'string',
                       'RoofMatl': 'string',
                       'Exterior1st': 'string',
                       'Exterior2nd': 'string',
                       'MasVnrType': 'string',
                       'ExterQual': 'string',
                       'ExterCond': 'string',
                       'Foundation': 'string',
                       'BsmtQual': 'string',
                       'BsmtCond': 'string',
                       'BsmtExposure': 'string',
                       'BsmtFinType1': 'string',
                       'BsmtFinType2': 'string',
                       'Heating': 'string',
                       'HeatingQC': 'string',
                       'CentralAir': 'string',
                       'Electrical': 'string',
                       'KitchenQual': 'string',
                       'Functional': 'string',
                       # 'FireplaceQu': 'string',
                       'GarageType': 'string',
                       'GarageFinish': 'string',
                       'GarageQual': 'string',
                       'GarageCond': 'string',
                       'PavedDrive': 'string',
                       # 'PoolQC': 'string',
                       # 'Fence': 'string',
                       # 'MiscFeature': 'string',
                       'MoSold': 'string',
                       'YrSold': 'string',
                       'SaleType': 'string',
                       'SaleCondition': 'string'
                       }

# Final features list to train the model
FINAL_FEATURES_LIST = list(NUMERICAL_COLUMNS.keys()) + list(CATEGORICAL_COLUMNS.keys()) + ['HouseAge',
                                                                                           'HouseAgeRemodel',
                                                                                           'GarageAge']

# Fields which should be provided by user/should be present in incoming requests
MANDATORY_INPUT_FIELDS = [
    'LotArea',
    'YearBuilt',
    'Foundation',
]
