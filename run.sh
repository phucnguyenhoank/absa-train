docker run \
    -e RUN_MODE="sanity_check" \
    -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/google_creds.json \
    -v $HOME/.config/gcloud/application_default_credentials.json:/tmp/keys/google_creds.json \
    absa-train:1.0
