FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310

# 1. Set the working directory
WORKDIR /app

# 2. Install system dependencies (Java, wget)
RUN apt-get update && apt-get install -y \
    default-jdk-headless \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN pip install --no-cache-dir \
    transformers==4.38.2 \
    datasets>=2.12.0 \
    py-vncorenlp>=0.1.4

# 4. Copy the rest of your code
COPY . .

# ENV TRUST_REMOTE_CODE=True
# ENV TORCH_SKIP_VERSION_CHECK=1 

# 5. The command that actually runs your script
CMD ["python", "train.py"]
