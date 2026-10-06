# ---- builder: produce the original model (v1) exactly as Milestone 3 does ----
# This becomes the "seed" model: on first start the backend copies it into the
# shared models folder as v1, and retraining builds v2, v3, ... next to it.
FROM python:3.13-slim AS builder
WORKDIR /build

COPY milestone_3/requirements.txt milestone_3/requirements.txt
RUN pip install --no-cache-dir -r milestone_3/requirements.txt

COPY training/bertweet_lora_final_fulldata training/bertweet_lora_final_fulldata
COPY milestone_3/export_onnx.py milestone_3/export_onnx.py

ENV HF_HOME=/build/.cache/huggingface
RUN python milestone_3/export_onnx.py

# ---- runtime: onnxruntime + transformers tokenizer only ----
FROM python:3.13-slim
WORKDIR /app

COPY milestone_4/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /build/milestone_3/onnx_model/model_int8.onnx ./seed_model/
COPY --from=builder /build/milestone_3/onnx_model/*.json /build/milestone_3/onnx_model/vocab.txt /build/milestone_3/onnx_model/bpe.codes ./seed_model/
COPY milestone_4/backend/app ./app

RUN useradd --create-home appuser && mkdir -p /models && chown -R appuser /app /models
USER appuser

ENV MODELS_DIR=/models SEED_MODEL_DIR=/app/seed_model

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
