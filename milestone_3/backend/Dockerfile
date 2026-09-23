# ---- builder: merge the LoRA adapter, export to ONNX, quantize to INT8 ----
# Kept separate from the runtime stage so the final image doesn't carry
# torch/peft/optimum (needed only to produce the ONNX file, not to serve it)
# or the intermediate FP32 PyTorch/ONNX weights (~1 GB) that export_onnx.py
# leaves behind alongside the ~136 MB INT8 model it actually produces.
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

COPY milestone_3/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /build/milestone_3/onnx_model ./onnx_model
COPY milestone_3/backend/app ./app

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
