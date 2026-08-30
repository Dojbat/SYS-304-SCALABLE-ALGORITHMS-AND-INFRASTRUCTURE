FROM python:3.13-slim

WORKDIR /app

COPY milestone_2/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY milestone_1/baseline_model.pkl .
COPY milestone_2/backend/app ./app

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
