FROM python:3.9.15

WORKDIR /app

COPY . .

RUN pip install -e .

# Runhouse server port
EXPOSE 50052
# Ray Redis cache port
EXPOSE 6379
# Ray dashboard port
EXPOSE 52365

CMD ["runhouse", "start", "--host", "0.0.0.0"]
