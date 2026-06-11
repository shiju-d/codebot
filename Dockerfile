FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config.py message.py runner.py ./

EXPOSE 8000

CMD ["uvicorn", "runner:app", "--host", "0.0.0.0", "--port", "8000"]
