FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 1. Set the working directory
WORKDIR /app

# 2. Install system dependencies (Java, wget)
RUN apt-get update && apt-get install -y \
    default-jdk-headless \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN pip install --no-cache-dir py-vncorenlp
RUN pip install --no-cache-dir transformers
RUN pip install --no-cache-dir datasets

# 4. Copy the rest of your code
COPY . .

# 5. The command that actually runs your script
# This is what always runs
ENTRYPOINT ["python", "train.py"]

# This is the default if the box is empty
CMD [] 
