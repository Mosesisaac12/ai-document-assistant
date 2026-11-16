FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY rag_app.py .
COPY rag_engine.py .
COPY evaluation_app.py .
COPY evaluation.py .

EXPOSE 8501 8502

CMD ["streamlit", "run", "rag_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
