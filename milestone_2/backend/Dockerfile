FROM python:3.13-slim

WORKDIR /app

COPY milestone_2/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/bertweet_lora_final_fulldata ./lora_model
COPY milestone_2/backend/app ./app

# Bake the base BERTweet weights into the image so the container doesn't need
# network access (or a re-download) every time it starts.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('vinai/bertweet-base', num_labels=2)"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
