How to start a local Docker container with public key based authentication

1. Configure Docker to use secrets in the build process

echo "DOCKER_BUILDKIT=1" >> ~/.docker/config.json

or edit the file manually to make sure it includes

{
  "features": {
    "buildkit": true
  }
}

2. Generate a public private key pair

mkdir -p ~/.ssh/runhouse/docker
ssh-keygen -t rsa -b 4096 -C "your_email@example.com" -f ~/.ssh/runhouse/docker/id_rsa

3. The Dockerfile in the current directory should support public key based authentication using Docker Secrets for its build process

4. Build the Docker container

docker build --no-cache --pull --rm -f "runhouse/docker/slim/public-key-auth/Dockerfile" --secret id=ssh_key,src=$HOME/.ssh/runhouse/docker/id_rsa.pub -t runhouse:start .

5. Run the Docker container

docker run --rm --shm-size=3gb -it -p 32300:32300 -p 6379:6379 -p 52365:52365 -p 22:22 -p 443:443 -p 80:80 runhouse:start

6. Verify via SSH

ssh -i ~/.ssh/runhouse/docker/id_rsa rh-docker-user@localhost
