from ast import List
from typing import Optional, Union

from runhouse.constants import DEFAULT_SERVER_PORT
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.hardware.utils import ServerConnectionType
from runhouse.utils import conda_env_cmd, venv_cmd

try:
    import kubernetes

    from kubernetes import client, config
    from kubernetes.stream import stream
except:
    pass


class PodsCluster(Cluster):
    RESOURCE_TYPE = "cluster"

    def __init__(
        self,
        # Name will almost always be provided unless a "local" cluster is created
        name: Optional[str] = None,
        kube_config_path: Optional[str] = None,
        namespace: str = None,
        pod_names: Union[str, List] = None,
        server_host: str = None,
        server_port: int = None,
        client_port: int = None,
        den_auth: bool = False,
        dryrun: bool = False,
        image: Optional["Image"] = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """PodsCluster"""
        super().__init__(
            name=name,
            ips=pod_names if isinstance(pod_names, list) else [pod_names],
            server_host=server_host or "0.0.0.0",
            server_port=server_port or DEFAULT_SERVER_PORT,
            server_connection_type=ServerConnectionType.SSH,
            client_port=client_port,
            den_auth=den_auth,
            image=image,
            dryrun=dryrun,
        )
        self.kube_config_path = kube_config_path
        self.namespace = namespace

    @property
    def pods(self):
        return self._ips

    def load_kube_config(self):
        if self.kube_config_path:
            config.load_kube_config(config_file=self.kube_config_path)
        else:
            config.load_kube_config()

    def config(self, condensed: bool = True):
        config = super().config(condensed)
        self.save_attrs_to_config(
            config,
            [
                "namespace",
            ],
        )
        return config

    def _ping(self, timeout=5, retry=False):
        self.load_kube_config()
        v1 = client.CoreV1Api()

        # TODO - run this in parallel?
        for pod_name in self.pods:
            try:
                pod = v1.read_namespaced_pod(name=pod_name, namespace=self.namespace)
                if pod.status.phase != "Running":
                    print(f"Pod {pod_name} is not running. Status: {pod.status.phase}")
                    return False
            except kubernetes.client.exceptions.ApiException as e:
                print(f"Error getting pod {pod_name}: {e}")
                return False
        return True

    def run_bash_over_ssh(
        self,
        commands,
        node=None,
        container=None,
        stream_logs=True,
        require_outputs=True,
        _ssh_mode="interactive",
        conda_env_name=None,
        venv_path=None,
    ):
        if node is None:
            node = self.head_ip

        if node == "all":
            res_list = []
            for node in self.ips:
                res = self.run_bash_over_ssh(
                    commands=commands,
                    stream_logs=stream_logs,
                    require_outputs=require_outputs,
                    node=node,
                    conda_env_name=conda_env_name,
                    venv_path=venv_path,
                )
                res_list.append(res)
            return res_list

        self.load_kube_config()
        api = client.CoreV1Api()

        venv_path = venv_path or self.image.venv_path if self.image else None
        if conda_env_name:
            commands = [conda_env_cmd(cmd, conda_env_name) for cmd in commands]
        if venv_path:
            commands = [venv_cmd(cmd, venv_path) for cmd in commands]

        if container is None:
            pod = api.read_namespaced_pod(name=node, namespace=self.namespace)
            if not pod.spec.containers:
                raise Exception(f"No containers found in pod {node}")
            container = pod.spec.containers[0].name

        commands = [["/bin/sh", "-c", command] for command in commands]

        for exec_command in commands:
            try:
                resp = stream(
                    api.connect_get_namespaced_pod_exec,
                    node,
                    self.namespace,
                    container=container,
                    command=exec_command,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                )

                # TODO - error handling. currently if the command runs and fails, exception is not thrown
                # but error is contained inside the resp

                return resp
            except Exception as e:
                raise Exception(
                    f"Failed to execute command {exec_command} on pod {node}: {str(e)}"
                )

    def rsync(
        self,
        source: str,
        dest: str,
        up: bool = True,
        node: str = None,
        src_node: str = None,
        contents: bool = False,
        filter_options: str = None,
        stream_logs: bool = False,
        ignore_existing: bool = False,
        parallel: bool = False,
    ):
        pass
