# HTTPServer with local span collection enabled
FROM python:3.9.15-slim-buster

WORKDIR /app

COPY . .

RUN pip install -e .

# Ray dashboard port
EXPOSE 52365
# Runhouse server port
EXPOSE 32300
# Ray Redis cache port
EXPOSE 6379
# HTTPS port
EXPOSE 443
# HTTP port
EXPOSE 80

CMD ["python", "-m", "runhouse.servers.http.http_server", "--enable_local_span_collection=True", "--host", "0.0.0.0", "--port", "32300"]
