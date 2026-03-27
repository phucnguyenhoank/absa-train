docker run \
    -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/google_creds.json \
    -v $HOME/.config/gcloud/application_default_credentials.json:/tmp/keys/google_creds.json \
    absa-train:1.0 \
    --output_dir=gs://absa-models-bucket/absa-training-sanity \
    --run_mode=sanity_check
