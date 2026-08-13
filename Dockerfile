FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

ARG PORT_DT_VERSION=3.2.0

LABEL org.opencontainers.image.title="Port Twin AI" \
      org.opencontainers.image.version="${PORT_DT_VERSION}" \
      org.opencontainers.image.description="Public-data offline port digital-twin and RL evidence runtime" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT_DT_ENV=development \
    PORT_DT_SERVER_PORT=8000

WORKDIR /opt/port-dt-multi
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 portdt

COPY --chown=portdt:portdt app ./app
COPY --chown=portdt:portdt data ./data
COPY --chown=portdt:portdt config ./config
COPY --chown=portdt:portdt evidence ./evidence
COPY --chown=portdt:portdt docs ./docs
COPY --chown=portdt:portdt scripts ./scripts
COPY --chown=portdt:portdt README.md LICENSE VERSION ./

# Selected checked-in run evidence is copied into the image. Keep writable
# runtime locations present for non-root training, reports and audit receipts
# without mutating the portable evidence bundle.
RUN mkdir -p data/objects data/rl/runs data/audit \
    && chown -R portdt:portdt data
USER portdt
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
