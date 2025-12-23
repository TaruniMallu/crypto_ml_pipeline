FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y libgomp1
RUN apt-get update && apt-get install -y libgomp1 gcc
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fix Python imports (src/, utils/, etc.)
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
