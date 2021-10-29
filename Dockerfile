FROM python:3.8.11-slim-buster as builder

WORKDIR /app

RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt > requirements.txt


FROM python:3.8.11-slim-buster

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=builder /app/requirements.txt .
RUN pip install -r requirements.txt

ARG PORT=8000
ENV PORT=${PORT}

WORKDIR /app
COPY . .

CMD ["gunicorn", "app:server", "-w", "1", "--reload", "--timeout", "300"]
