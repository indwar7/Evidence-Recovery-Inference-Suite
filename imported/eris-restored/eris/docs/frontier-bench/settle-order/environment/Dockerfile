FROM python:3.13-slim-bookworm

# pytest so scratch checks can be written under /app.
RUN pip install --no-cache-dir pytest==9.1.1

WORKDIR /app

COPY src/ /app/src/
COPY harness.py /app/harness.py
COPY README.md /app/README.md
