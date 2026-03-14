FROM python:3.12-slim

# 1. Set the working directory
WORKDIR /app

# 2. Install system dependencies (Java, wget)
RUN apt-get update && apt-get install -y \
    default-jdk-headless \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# 3. Copy requirements first (better for caching)
RUN pip install --no-cache-dir torch>=2.10.0

RUN pip install --no-cache-dir transformers>=5.3.0 datasets>=4.7.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your code
COPY . .

# 5. The command that actually runs your script
CMD ["python", "train.py"]
