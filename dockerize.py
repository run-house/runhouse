# def local_docker_cluster():
#     port = 6379
#     # Check if container is already running
#     import docker
#     import runhouse as rh
#     import os
#     client = docker.from_env()
#     if "rh_cluster" not in [c.name for c in client.containers.list(filters={"status": "running"})]:
#         client.images.pull("berkeleyskypilot/skypilot:latest")
#         print("got here")
#         from pathlib import Path
#         import pkgutil
#         print("got here now ")
#         runhouse_path = Path(pkgutil.get_loader("runhouse").path).parent.parent
#         if "rh_cluster" in [c.name for c in client.containers.list(all=True)]:
#             client.containers.get("rh_cluster").remove()
#             print("in here")
#         client.containers.run("berkeleyskypilot/skypilot:latest",
#                               command="tail -f /dev/null",
#                               name="rh_cluster",
#                               detach=True,
#                               shm_size="2.15gb",
#                               auto_remove=True,
#                               ports={port: port},
#                               volumes={os.path.expanduser("~/.sky"): {"bind": "/root/.sky", "mode": "rw"},
#                                        runhouse_path: {"bind": "/root/runhouse", "mode": "rw"}
#                                        }
#                               )
#     c = rh.cluster(name="local-docker", host=f"localhost:{port}", ssh_creds={"ssh_user": "root", "ssh_proxy_command": "docker exec -it rh_cluster /bin/bash"})
#     c.up_if_not()
#     c.install_packages(["pytest"])
#     return c

# local_docker_cluster()

import runhouse as rh
port = 50052
c = rh.cluster(name="local-docker", host=f"localhost:{port}", ssh_creds={"ssh_user": "root", "ssh_proxy_command": "docker exec -it ro-http-server-container /bin/bash"})
c.up_if_not()
print(c.is_up())
