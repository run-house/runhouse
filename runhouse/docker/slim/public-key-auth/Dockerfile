# Debian based image (should work for Ubuntu as well) using public key authentication
FROM python:3.9.15-slim

ARG RUNHOUSE_PATH
ARG RUNHOUSE_VERSION

WORKDIR /app

COPY . .

# Create the password file directory
RUN mkdir -p /app/ssh

# Install the required packages
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --no-install-recommends gcc python3-dev openssh-server rsync supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# COPY local Runhouse package into the image if provided
COPY $RUNHOUSE_PATH /app/runhouse

# If using a local version of runhouse, install it from the local directory
RUN if [ -d "/app/runhouse" ]; then pip install -U -e /app/runhouse; else pip install -U runhouse[opentelemetry]==$RUNHOUSE_VERSION; fi

# Create the privilege separation directory required by sshd
RUN mkdir -p /run/sshd

# Create a user for SSH access
RUN useradd -m rh-docker-user

# Create .ssh directory for the user
RUN mkdir -p /home/rh-docker-user/.ssh

# Copy the public key into the image using Docker Build Secrets
RUN --mount=type=secret,id=ssh_key,dst=/tmp/id_rsa.pub \
    cp /tmp/id_rsa.pub /home/rh-docker-user/.ssh/authorized_keys && \
    chown -R rh-docker-user:rh-docker-user /home/rh-docker-user/.ssh && \
    chmod 700 /home/rh-docker-user/.ssh && \
    chmod 600 /home/rh-docker-user/.ssh/authorized_keys

# Update SSHD Configuration to allow public key authentication and disable password and root login
RUN echo "PasswordAuthentication no" >> /etc/ssh/sshd_config && \
    echo "PermitRootLogin no" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config

# Create supervisord configuration file
RUN echo "[supervisord]" > /etc/supervisor/conf.d/supervisord.conf && \
    echo "nodaemon=true" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "user=root" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:sshd]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=/usr/sbin/sshd -D" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile=/var/log/sshd.log" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile=/var/log/sshd.err" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:runhouse]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=runhouse restart --no-screen --host "0.0.0.0"" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile=/var/log/runhouse.log" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile=/var/log/runhouse.err" >> /etc/supervisor/conf.d/supervisord.conf

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
# SSH port
EXPOSE 22

# Run supervisord as the main process to manage the others
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
