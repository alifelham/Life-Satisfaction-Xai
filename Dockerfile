FROM python:3.10.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-paper.txt pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-paper.txt \
    && python -m pip install --no-deps -e .

COPY . .

ENTRYPOINT ["python", "-m", "life_satisfaction.pipeline"]
CMD ["--help"]
