# Debian based image (should work for Ubuntu as well)
FROM python:3.9.15-slim

ARG DOCKER_USER_PASSWORD_FILE

WORKDIR /app

COPY . .

# Create the password file directory
RUN mkdir -p /app/ssh

# Install the required packages
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --no-install-recommends gcc python3-dev openssh-server rsync supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy the password file into the image
# TODO(denis): check if the password file doesn't exist, if that's the case,
# generate a new password and save it to the file.
COPY $DOCKER_USER_PASSWORD_FILE /app/ssh/docker_user_password_file

RUN pip install -e .

# Create the privilege separation directory required by sshd
RUN mkdir -p /run/sshd

# Create a user for SSH access (using password from $DOCKER_USER_PASSWORD_FILE)
RUN useradd -m rh-docker-user && \
    echo "rh-docker-user:$(cat /app/ssh/docker_user_password_file)" | chpasswd && \
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
