FROM otel/opentelemetry-collector-contrib:latest

# Define an argument for the config file path
ARG CONFIG_FILE=otel-collector-config.yaml

# Copy the config file into the container
COPY ${CONFIG_FILE} /otel-collector-config.yaml

# Expose the necessary ports (health check, grpc, and http ports)
EXPOSE 13134 4319 4316

# Set the command to run the collector with the config file
CMD ["--config", "/otel-collector-config.yaml"]
