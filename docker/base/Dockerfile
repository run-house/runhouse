# Use the latest Miniconda3 image with Python 3.10
FROM continuumio/miniconda3:23.3.1-0

# Build arguments to determine installation source and runhouse version to install
ARG INSTALL_METHOD=pypi
ARG RUNHOUSE_VERSION

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    rsync \
    vim \
    gcc \
    screen \
    wget \
    curl \
    sudo \
    ufw \
    git \
    supervisor \
    awscli && \
    rm -rf /var/lib/apt/lists/*

# Copy top-level files to /app/
COPY setup.py pyproject.toml README.md requirements.txt pytest.ini /app/

# Copy the runhouse directory that contains the app code to /app/runhouse/
COPY runhouse/ /app/runhouse/

# Determine which version of runhouse to install
RUN if [ "$INSTALL_METHOD" = "local" ] && [ -d "/app/runhouse" ] && [ "$(ls -A /app/runhouse)" ]; then \
      echo "Installing runhouse from local"; \
      # Install the runhouse package with the sky dependency
      pip install --no-cache-dir /app"[sky]"; \
    else \
      echo "Installing runhouse from PyPI"; \
      if [ -z "$RUNHOUSE_VERSION" ] || [ "$RUNHOUSE_VERSION" = "latest" ]; then \
        echo "Installing latest version"; \
        pip install --no-cache-dir "runhouse[sky]"; \
      else \
        echo "Installing runhouse==${RUNHOUSE_VERSION}"; \
        pip install --no-cache-dir "runhouse[sky]==${RUNHOUSE_VERSION}"; \
      fi; \
    fi

# Create the privilege separation directory required by sshd
RUN mkdir -p /run/sshd

# Create supervisord configuration file
RUN echo "[supervisord]" > /etc/supervisor/conf.d/supervisord.conf && \
    echo "nodaemon=true" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "user=root" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:sshd]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=/usr/sbin/sshd -D" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile=/var/log/sshd.log" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile=/var/log/sshd.err" >> /etc/supervisor/conf.d/supervisord.conf

# Keep the container running with supervisord as the main process
# This will allow SkyPilot to connect and run commands inside the container
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
