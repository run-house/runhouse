# Prerequisites: Start a local Docker image with Runhouse installed using the following commands:
#   $ docker build --pull --rm -f \"Dockerfile\" \
#       --build-arg DOCKER_USER_PASSWORD_FILE=/etc/ssh/docker_user_passwd \
#       -t runhouse:start .
#   $ docker run --rm --shm-size=3gb -it -p 50052:50052 -p 6379:6379 -p 52365:52365 -p 22:22 \
#       runhouse:start
print("Starting Runhouse in a local Docker cluster")
import runhouse as rh

print("Runhouse version: ", rh.__version__)
rh.configs.disable_data_collection()  # Workaround until we remove the usage of GCSClient from our code
port = 50052
c = rh.cluster(
    name="local-docker", host=f"localhost:{port}", ssh_creds={"ssh_user": "root"}
)
print("Cluster created")
c.up_if_not()
print("Cluster is up: ", c.is_up())
c.check_server()
c.run(["pwd"])
print("Putting key my_key with value 7")
c.put("my_key", "7")
print("Getting my_key, value is: " + c.get("my_key"))
