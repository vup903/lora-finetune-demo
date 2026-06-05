# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY . .

# Default: run a LoRA fine-tune on CPU (distilgpt2). Override the command to use
# a different model, or run generate.py.
ENTRYPOINT ["python", "train.py"]
