FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    numpy

WORKDIR /app
