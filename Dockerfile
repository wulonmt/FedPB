# ==== Base image ====
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# ==== Set working directory ====
WORKDIR /workspace

# ==== Copy project files ====
COPY . .

# ==== Install dependencies ====
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get install g++

# Upgrade pip and install requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \

# ==== Environment variables ====
ENV PYTHONUNBUFFERED=1
ENV RAY_DEDUP_LOGS=0

# ==== Default command ====
# CMD ["bash"]
