FROM python:3.10-slim

RUN pip install --no-cache-dir jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install --no-cache-dir numpy

WORKDIR /app
COPY evaluate.py .

ENTRYPOINT ["python", "evaluate.py"]
