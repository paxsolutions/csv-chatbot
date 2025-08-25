FROM python:3.9-slim

WORKDIR /src

EXPOSE 8501

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV LANG C.UTF-8

COPY . /src

RUN pip3 install --no-cache-dir --upgrade pip && pip3 install -r requirements.txt --no-cache-dir

ENTRYPOINT ["streamlit", "run", "src/chatbot_csv.py", "--server.port=8501"]