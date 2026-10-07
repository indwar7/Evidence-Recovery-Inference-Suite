FROM python:3.13-slim-bookworm

# runuser (util-linux) drops privileges before any submitted code runs.
RUN apt-get update && apt-get install -y --no-install-recommends util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pytest==9.1.1 pytest-json-ctrf==0.5.2

# Unprivileged account that runs the submitted store.
RUN useradd -m runner

# Bake the sealed tests, then lock them to root. The account that runs submitted
# code can never read the reference, the generator, or the expected answers.
COPY . /tests/
RUN chown -R root:root /tests && chmod -R 700 /tests

# Pre-create the artifact parent so the harness can place /app/src/store.py.
RUN mkdir -p /app/src
