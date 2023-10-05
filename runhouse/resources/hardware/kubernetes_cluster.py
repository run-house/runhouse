import logging
import os
import subprocess
from pathlib import Path
from typing import Union

import requests
import yaml

from runhouse.globals import rns_client
from runhouse.resources.hardware.on_demand_cluster import OnDemandCluster

logger = logging.getLogger(__name__)


class KubernetesCluster(OnDemandCluster):
    def __init__(
        self,
        name,
        instance_type: str = None,
        num_instances: int = None,
        provider: str = None,
        dryrun=False,
        autostop_mins=None,
        use_spot=False,
        image_id=None,
        region=None,
        sky_state=None,
        namespace: str = "default",
        num_workers: int = None,
        cpus: int = None,
        memory: int = None,
        num_gpus: int = None,
        head_cpus: int = None,
        head_memory: int = None,
        **kwargs,
    ):
        # To be initialized when upping the codeflare-sdk cluster
        self.cluster = None

        super().__init__(
            name,
            instance_type,
            num_instances,
            provider,
            dryrun,
            autostop_mins,
            use_spot,
            image_id,
            region,
            sky_state,
            **kwargs,
        )

        self.namespace = namespace
        self.num_workers = num_workers
        self.cpus = cpus
        self.memory = memory
        self.num_gpus = num_gpus
        self.head_cpus = head_cpus
        self.head_memory = head_memory

    @property
    def _mcad_cluster_role_path(self):
        return str(
            Path(__file__).parent / "k8s" / "mcad-controller-ray-clusterrole.yaml"
        )

    @property
    def _mcad_cluster_binding_path(self):
        return str(
            Path(__file__).parent
            / "k8s"
            / "mcad-controller-ray-clusterrolebinding.yaml"
        )

    def is_up(self) -> bool:
        """Whether the cluster is up.

        Example:
            >>> rh.kubernetes_cluster("rh-k8s").is_up()
        """
        status = self.status()
        return status == "READY" if status is not None else False

    def status(self, refresh: bool = True) -> Union[str, None]:
        """Status of the cluster.

        Example:
            >>> rh.kubernetes_cluster("rh-k8s").status()
        """
        if self.cluster is None:
            return None

        # Returns the requested cluster's status, as well as whether or not it is ready for use.
        status, ready = self.cluster.status()
        return status.name

    def teardown(self):
        """Teardown cluster.

        Example:
            >>> rh.kubernetes_cluster("rh-k8s").teardown()
        """
        if self.cluster:
            # Deletes the AppWrapper yaml, scaling-down and deleting all resources associated with the cluster.
            self.cluster.down()

    def teardown_and_delete(self):
        """Teardown cluster and delete it from configs.

        Example:
            >>> rh.kubernetes_cluster("rh-k8s").teardown_and_delete()
        """
        # TODO [RS] delete from kube config? or anywhere else besides Den that the codeflare-sdk would not handle?
        self.teardown()
        rns_client.delete_configs()

    def up(self):
        """Up the cluster.

        Example:
            >>> rh.kubernetes_cluster(
            >>>     name="cpu-cluster-test",
            >>>     instance_type="CPU:1",
            >>>     provider="kubernetes",
            >>> )
        """
        if self.on_this_cluster():
            return self

        # TODO [RS] will this ever not be kubernetes?
        if self.provider in ["kubernetes"]:

            # setup EKS K8s cluster for codeflare sdk usage

            # Install Kuberay Operator (Ensure helm is installed locally)
            add_kuberay = (
                "helm repo add kuberay https://ray-project.github.io/kuberay-helm/"
            )
            logger.info(f"Running {add_kuberay}")
            res = subprocess.run(
                add_kuberay, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            # Note: if receiving warning about kube config permissions, restrict r/w to
            # owner only: ``chmod 600 ~/.kube/config``
            if res.returncode != 0:
                raise Exception(res.stderr)

            kuberay_operator = "helm list --filter kuberay-operator"
            res = subprocess.run(
                kuberay_operator,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if res.returncode == 0 and "kuberay-operator" in res.stdout:
                # Note: if kuberay-operator already in use, can also run: ``helm uninstall kuberay-operator``
                logger.info("Kuberay operator already installed.")
            else:
                install_kuberay = (
                    "helm install kuberay-operator kuberay/kuberay-operator"
                )
                logger.info(f"Running {install_kuberay}")
                res = subprocess.run(
                    install_kuberay,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if res.returncode != 0:
                    raise Exception(res.stderr)

            repo_dir = "codeflare-operator"
            if not os.path.exists(repo_dir):
                # Install Codeflare Operator (Ensure gnu sed is installed locally)
                clone_cf_operator = (
                    "git clone https://github.com/RohanSreerama5/codeflare-operator.git"
                )
                cd_cmd = f"cd {repo_dir}"
                make_install = (
                    "make install -e SED=/opt/homebrew/opt/gnu-sed/libexec/gnubin/sed"
                )
                make_deploy = (
                    "make deploy -e SED=/opt/homebrew/opt/gnu-sed/libexec/gnubin/sed"
                )

                logger.info(
                    f"Running {clone_cf_operator} && {cd_cmd} && {make_install} && {make_deploy}"
                )

                cmd = f"{clone_cf_operator} && {cd_cmd} && {make_install} && {make_deploy}"
                res = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if res.returncode != 0:
                    raise Exception(res.stderr)

            # Setup permissions so Codeflare Operator can manage Ray clusters (Ensure kubectl is setup locally)
            perm_1 = f"kubectl apply -f {self._mcad_cluster_role_path}"
            perm_2 = f"kubectl apply -f {self._mcad_cluster_binding_path}"

            logger.info(f"Running {perm_1} && {perm_2}")

            cmd = f"{perm_1} && {perm_2}"
            res = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if res.returncode != 0:
                raise Exception(res.stderr)

            # Clone down the codeflare-sdk. We cannot get it from PyPi bc we make changes in the SDK to enable
            # non-Openshift K8s support
            if not os.path.exists("codeflare-sdk"):
                clone_cf = (
                    "git clone https://github.com/RohanSreerama5/codeflare-sdk.git"
                )

                logger.info(f"Running {clone_cf}")

                cmd = f"{clone_cf}"
                res = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if res.returncode != 0:
                    raise Exception(res.stderr)

            # Import pieces from codeflare-sdk
            try:
                from codeflare_sdk.cluster.auth import KubeConfigFileAuthentication
                from codeflare_sdk.cluster.cluster import Cluster, ClusterConfiguration
            except ImportError:
                raise ImportError(
                    "codeflare_sdk required for Kubernetes clusters. Install Kubernetes dependencies "
                    "with: `pip install runhouse[kubernetes]`"
                )

            # Now we can deploy user Ray clusters
            kube_config_path = os.path.expanduser("~/.kube/config")
            auth = KubeConfigFileAuthentication(kube_config_path=kube_config_path)
            auth.load_kube_config()

            # Create and configure our cluster object (and appwrapper)
            namespace = self.namespace
            cluster_name = self.name
            local_interactive = True

            self.cluster = Cluster(
                ClusterConfiguration(
                    local_interactive=local_interactive,
                    name=cluster_name,
                    namespace=namespace,
                    num_workers=self.num_workers,
                    min_cpus=self.cpus,
                    max_cpus=self.cpus,
                    min_memory=self.memory,
                    max_memory=self.memory,
                    num_gpus=self.num_gpus or 0,
                    # needs to be set properly to ensure Ray version. currently Ray version is 2.7.0.
                    # needs to match client(local). This here is server's version
                    image="rayproject/ray:2.6.3",
                    instascale=False,
                    # TODO [RS] which version of the codeflare-sdk are these attributes present?
                    # head_cpus=self.head_cpus,
                    # head_memory=self.head_memory
                    # mcad=False
                )
            )

            yaml_file_path = f"{cluster_name}.yaml"
            with open(yaml_file_path, "r") as file:
                yaml_dict = yaml.safe_load(file)

            for item in yaml_dict["spec"]["resources"]["GenericItems"][:]:
                if (
                    "generictemplate" in item
                    and item["generictemplate"]["apiVersion"] == "route.openshift.io/v1"
                ):
                    yaml_dict["spec"]["resources"]["GenericItems"].remove(item)

            modified_yaml_str = yaml.dump(yaml_dict, default_flow_style=False)

            with open(yaml_file_path, "w") as file:
                file.write(modified_yaml_str)

            # up the codeflare ray cluster
            if not self.is_up():
                self.cluster.up()

            try:
                # TODO [RS] in codeflare-sdk v0.8.0 no option to disable dashboard check?
                # cluster.wait_ready(dashboard_check=False)
                self.cluster.wait_ready()
            except requests.exceptions.MissingSchema as e:
                logger.warning(e)
                pass

        else:
            raise ValueError(f"Cluster provider {self.provider} not supported.")

        self.restart_server(restart_ray=True)

        return self

    def restart_server(
        self,
        _rh_install_url: str = None,
        resync_rh: bool = True,
        restart_ray: bool = True,
        env_activate_cmd: str = None,
    ):
        """Restart the RPC server.

        Args:
            resync_rh (bool): Whether to resync runhouse. (Default: ``True``)
            restart_ray (bool): Whether to restart Ray. (Default: ``True``)
            env_activate_cmd (str, optional): Command to activate the environment on the server. (Default: ``None``)
        Example:
            >>> rh.kubernetes_cluster("rh-cpu").restart_server()
        """
        logger.info(f"Restarting HTTP server on {self.name}.")

        if resync_rh:
            # consider changing this to use kubectl get rayclusters instead
            command = f"kubectl get pods -n {self.namespace} | grep {self.name}-head"
            output = subprocess.run(
                command,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            lines = output.stdout.split("\n")
            if len(lines) <= 1:
                raise RuntimeError(
                    f"No matching pods found in namespace: `{self.namespace}` and name: `{self.name}`."
                )

            pod_name = lines[1].split()[0]
            logger.info(f"Found matching pod: {pod_name}")

            kubectl_command = (
                f"kubectl get pod {pod_name} -n default -o "
                f"jsonpath='{{.status.podIP}}' | awk '{{print $1}}'"
            )

            logger.info(f"Running {kubectl_command}")
            result = subprocess.run(
                kubectl_command,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                raise Exception(result.stderr)

            pod_ip = result.stdout
            logger.info(f"Pod IP Address: {pod_ip}")
            self.address = pod_ip

            # sync_rh_cmd = f"kubectl cp ../../../../runhouse/ default/{pod_name}:runhouse"
            sync_rh_cmd = f"kubectl cp ../runhouse/ default/{pod_name}:runhouse"
            result = subprocess.run(
                sync_rh_cmd,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                raise Exception(result.stderr)

        install_rh_cmd = "pip install ./runhouse"

        cmd_1 = self.CLI_RESTART_CMD + (" --no-restart-ray" if not restart_ray else "")
        cmd_1 = f"{install_rh_cmd} && {cmd_1}"

        cmd = f"kubectl exec -n {self.namespace} {pod_name} -- {cmd_1}"

        logger.info(f"Running command on {self.name}: {cmd}")

        process = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.returncode != 0:
            raise Exception(process.stderr)

        logger.info(f"Command Output on {self.name}:")
        logger.info(process.stdout)
