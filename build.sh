#!/bin/bash

# uv export --format requirements-txt --output-file requirements.txt -q

# docker rmi asia-southeast1-docker.pkg.dev/ute-nlp-absa/absa-repo/absa-train:1.0
docker rm -f $(docker ps -a -q --filter ancestor=absa-train:1.0)
docker rmi absa-train:1.0
docker build --tag absa-train:1.0 .
