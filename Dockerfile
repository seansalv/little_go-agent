FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY go-agent-api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY go-agent-api /app/go-agent-api
COPY my_player3.py /app/my_player3.py

ENV PYTHONPATH="/app/go-agent-api:${PYTHONPATH}" \
	PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn go_agent_api.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

