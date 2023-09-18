
FROM python:3.9

WORKDIR /app

# COPY requirements.txt .

COPY . .

RUN pip install -e .

EXPOSE 50052

# CMD ["python", "-m", "runhouse.servers.http.http_server"]
CMD ["runhouse", "start"]
