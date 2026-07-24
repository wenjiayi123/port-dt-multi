FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT_DT_ENV=development \
    PORT_DT_SERVER_PORT=8000

WORKDIR /opt/port-dt-multi
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 portdt

COPY --chown=portdt:portdt app ./app
COPY --chown=portdt:portdt data ./data
COPY --chown=portdt:portdt config ./config
USER portdt
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
