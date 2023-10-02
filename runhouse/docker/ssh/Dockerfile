# Use the specified Python base image
FROM python:3.9.15

WORKDIR /app

COPY . .

RUN pip install -e .

# Create the privilege separation directory required by sshd
RUN mkdir -p /run/sshd

# Install SSH server and supervisord
RUN apt-get update && \
    apt-get install -y openssh-server supervisor && \
    rm -rf /var/lib/apt/lists/*

# Create a user for SSH access (using password: 123456)
RUN useradd -m rh-docker-user && \
    echo "rh-docker-user:123456" | chpasswd && \
    echo "PermitRootLogin no" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

# Create supervisord configuration file
RUN echo "[supervisord]" > /etc/supervisor/conf.d/supervisord.conf && \
    echo "nodaemon=true" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "user=root" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:sshd]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=/usr/sbin/sshd -D" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile=/var/log/sshd.log" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile=/var/log/sshd.err" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:runhouse]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=runhouse start --host "0.0.0.0"" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile=/var/log/runhouse.log" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile=/var/log/runhouse.err" >> /etc/supervisor/conf.d/supervisord.conf

# Runhouse server port
EXPOSE 50052
# Ray Redis cache port
EXPOSE 6379
# Ray dashboard port
EXPOSE 52365
# SSH port
EXPOSE 22

# Run supervisord as the main process to manage the others
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
