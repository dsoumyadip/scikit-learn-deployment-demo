import os

from google.cloud import storage

from demo.constants import BUCKET_NAME, TRAIN_FILE_NAME, DATA_DIR, MODEL_FILE_NAME, MODEL_DIR, \
    PREPROCESSING_PIPELINE_FILE_NAME


def get_train_data() -> None:
    """Fetches train files from cloud storage
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_train = bucket.blob(TRAIN_FILE_NAME)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    blob_train.download_to_filename(os.path.join(DATA_DIR, TRAIN_FILE_NAME))


def upload_files(source_file_path: str, filename: str) -> None:
    """
    Upload files to cloud storage
    Args:
        source_file_path: Path of the file
        filename: Filename to set in cloud storage

    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.upload_from_filename(source_file_path)


def get_model_and_preprocessing_pipeline() -> None:
    """Fetches latest model and preprocessing pipeline from cloud storage
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_model = bucket.blob(MODEL_FILE_NAME)
    blob_model.download_to_filename(os.path.join(MODEL_DIR, MODEL_FILE_NAME))
    blob_pipeline = bucket.blob(PREPROCESSING_PIPELINE_FILE_NAME)
    blob_pipeline.download_to_filename(os.path.join(MODEL_DIR, PREPROCESSING_PIPELINE_FILE_NAME))
