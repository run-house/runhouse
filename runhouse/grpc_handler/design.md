# GRPC for Send Execution

### Quick links:
* https://grpc.io/docs/languages/python/basics/
* Security and ports
  * [Ray ports](https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations)
  * [GRPC auth](https://grpc.io/docs/guides/auth/)
  * [More GRPC auth](https://grpc.github.io/grpc/python/grpc.html#authentication-authorization-objects)
  * As of right now we're just using port forwarding through [Paramiko and Plumber](https://plumbum.readthedocs.io/en/latest/remote.html#tunneling-example).