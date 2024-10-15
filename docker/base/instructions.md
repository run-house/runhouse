## Runhouse base image
You can use the Runhouse base image to run your own code in a containerized environment, or use as the base image
when launching an on-demand cluster, which includes Sky and Ray already installed.

The complete list of supported images can be found in the
[Container Registry](https://github.com/run-house/runhouse/pkgs/container/runhouse).

### Build the base image

#### Using local Runhouse code (for development)

*Note*: adjust the platform based on your host's architecture.

```bash
docker build --platform linux/amd64 -t runhouse:latest --build-arg INSTALL_METHOD=local -f docker/base/Dockerfile .
```
#### Using a specific Runhouse version

```bash
docker build --platform linux/amd64 -t runhouse:latest --build-arg RUNHOUSE_VERSION=0.0.35 -f docker/base/Dockerfile .
```

### Running the container
```bash
docker run -it runhouse:latest /bin/bash
```

### Launching a cluster with the image

```python
import runhouse as rh

cluster = rh.ondemand_cluster(name="my-cpu-cluster",
                              instance_type="CPU:2+",
                              provider="aws",
                              image_id="docker:ghcr.io/run-house/runhouse:latest").up_if_not()
