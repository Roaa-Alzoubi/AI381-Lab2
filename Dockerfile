
FROM python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60 AS builder
RUN python -m pip install --no-cache-dir --upgrade \
    "pip>=26.2.2" "setuptools>=84.2.1" "wheel>=0.48.0"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade \
       "setuptools>=78.1.1" \
       "wheel>=0.46.2" \
       "urllib3>=2.8.0" \
       "msgpack>=1.2.1" \
       "jaraco.context>=6.1.0"

FROM python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60 AS runtime
RUN python -m pip install --no-cache-dir --upgrade \
    "pip>=26.2.2" "setuptools>=84.2.1" "wheel>=0.48.0" \
    && rm -rf /usr/local/lib/python3.11/site-packages/setuptools-79.0.1.dist-info \
              /usr/local/lib/python3.11/site-packages/setuptools
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY model.joblib ./model.joblib

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
