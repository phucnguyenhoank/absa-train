from google.cloud import storage


def upload_blob(source_file, destination_path):
    # 1. Create the client (the connection)
    client = storage.Client(project="ute-nlp-absa")

    # 2. Link the path to the client
    # This tells Google: "Use this connection to find this file"
    blob = storage.Blob.from_string(
        f"{destination_path}/{source_file}", client=client
    )

    # 3. Upload
    blob.upload_from_filename(source_file)

    print(f"Done! {source_file} is now in the cloud.")


if __name__ == "__main__":
    upload_blob(
        "loss_history.json", "gs://absa-models-bucket/absa-training-cde"
    )
