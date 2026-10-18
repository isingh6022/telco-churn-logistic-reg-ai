# make build && make run

FROM python:3.14-slim

# make defaults explicity.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# dependencies - built into image itself - to avoid installation.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# only what api.py imports, plus the trained model
# skiped - model training files.
COPY api.py ./
COPY model/__init__.py model/all_names.py model/models.py model/
COPY model_output/churn_model.pkl model_output/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080

# useful for other services - and docker compose conditions.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
