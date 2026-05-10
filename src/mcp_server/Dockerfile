FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .

RUN uv pip install --system --no-cache-dir -r pyproject.toml

COPY . .

ENTRYPOINT ["python", "main.py"]