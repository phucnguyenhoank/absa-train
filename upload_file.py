from google.cloud import storage


def upload_model(
    source_file_path,
    destination_blob_name=None,
    bucket_name="absa-model-storage",
):
    """Uploads a trained model to an existing GCS bucket."""

    if not destination_blob_name:
        destination_blob_name = f"models/{source_file_path}"

    # Initialize the GCS client
    storage_client = storage.Client(project="ute-nlp-absa")

    # Get the existing bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a blob object (the file's destination in the bucket)
    blob = bucket.blob(destination_blob_name)

    print(f"Uploading {source_file_path}...")
    # This automatically uses resumable uploads for your 500MB file
    blob.upload_from_filename(source_file_path)

    print(
        f"Success! Model saved to gs://{bucket_name}/{destination_blob_name}"
    )


if __name__ == "__main__":
    LOCAL_MODEL = "model.txt"
    upload_model(LOCAL_MODEL)
