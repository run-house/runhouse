# import sky
from pathlib import Path
import subprocess
import logging
import time
import os

import yaml

# from runhouse.resources.hardware.on_demand_cluster import OnDemandCluster
from runhouse.resources.hardware.cluster import Cluster as cluser
from runhouse.servers.http import DEFAULT_SERVER_PORT, HTTPClient

# from typing import Optional

logger = logging.getLogger(__name__)

class KubernetesCluster(cluser):

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
            upped_cluster: bool = False,
            cluster_obj = None,
            **kwargs
    ):
        
        # super().__init__(
        #     name, 
        #     instance_type, 
        #     num_instances, 
        #     provider, 
        #     dryrun, 
        #     autostop_mins, 
        #     use_spot, 
        #     image_id, 
        #     region, 
        #     sky_state, 
        #     **kwargs
        # )

        

        self.namespace = namespace
        self.num_workers = num_workers
        self.cpus = cpus
        self.memory = memory
        self.num_gpus = num_gpus
        self.head_cpus = head_cpus
        self.head_memory = head_memory
        self.upped_cluster = upped_cluster
        self.provider = provider

        

        super().__init__(name=name, ips=[self.namespace], ssh_creds={}, dryrun=dryrun)

        check_pwd = f"pwd"
        result = subprocess.run("pwd", shell=True, check=True, stdout=subprocess.PIPE, text=True)
    
        # Get the captured output as a string
        pwd_output = result.stdout

        if "ray" not in pwd_output:

            install_cf = f"cd runhouse/resources/hardware/codeflare-sdk && pip install -e ."

            logger.info(f"Running {install_cf}")


            cmd = f"{install_cf}"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")


            from codeflare_sdk.cluster.cluster import Cluster, ClusterConfiguration
            from codeflare_sdk.cluster.auth import KubeConfigFileAuthentication


            kube_config_path = os.path.expanduser("~/.kube/config")
            auth = KubeConfigFileAuthentication(
                kube_config_path = kube_config_path
            )
            auth.load_kube_config()

            # Create and configure our cluster object (and appwrapper)
            namespace = self.namespace
            cluster_name = self.name
            local_interactive = True

            cluster = Cluster(ClusterConfiguration(
                local_interactive=local_interactive,
                name=cluster_name,
                namespace=namespace,
                num_workers=self.num_workers,
                min_cpus=self.cpus,
                max_cpus=self.cpus,
                min_memory=self.memory,
                max_memory=self.memory,
                num_gpus=self.num_gpus,
                image="rayproject/ray:2.6.3", # needs to be set properly to ensure Ray version. currently Ray version is 2.7.0. needs to match client(local). This here is server's version
                instascale=False,
                head_cpus=self.head_cpus,
                head_memory=self.head_memory # don't need to surface head_node specs to user for now 
                    # mcad=False 
            ))

            self.cluster_obj = cluster



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

        if self.provider in ["kubernetes"]:

            # setup EKS K8s cluster for codeflare sdk usage

            # Install Kuberay Operator (Ensure helm is installed locally)
            # add_kuberay = f"helm repo add kuberay https://ray-project.github.io/kuberay-helm/"
            # logger.info(f"Running {add_kuberay}")
            # cmd = f"{add_kuberay}"
            # try:
            #     subprocess.run(cmd, shell=True, check=True)
            # except subprocess.CalledProcessError as e:
            #     print(f"Error: {e}")

            # install_kuberay = f"helm install kuberay-operator kuberay/kuberay-operator"
            # logger.info(f"Running {install_kuberay}")
            # cmd = f"{install_kuberay}"
            # try:
            #     subprocess.run(cmd, shell=True, check=True)
            # except subprocess.CalledProcessError as e:
            #     print(f"Error: {e}")

            # # Install Codeflare Operator (Ensure gnu sed is installed locally)
            # clone_cf_operator = f"git clone git@github.com:RohanSreerama5/codeflare-operator.git"
            # cd_cmd = f"cd codeflare-operator"
            # make_install = f"make install -e SED=/opt/homebrew/opt/gnu-sed/libexec/gnubin/sed"
            # make_deploy = f"make deploy -e SED=/opt/homebrew/opt/gnu-sed/libexec/gnubin/sed"

            # logger.info(f"Running {clone_cf_operator} && {cd_cmd} && {make_install} && {make_deploy}")

            # cmd = f"{clone_cf_operator} && {cd_cmd} && {make_install} && {make_deploy}"
            # try:
            #     subprocess.run(cmd, shell=True, check=True)
            # except subprocess.CalledProcessError as e:
            #     print(f"Error: {e}")

            # # Setup permissions so Codeflare Operator can manage Ray clusters (Ensure kubectl is setup locally)
            # perm_1 = f"kubectl apply -f mcad-controller-ray-clusterrole.yaml"
            # perm_2 = f"kubectl apply -f mcad-controller-ray-clusterrolebinding.yaml"

            # logger.info(f"Running {perm_1} && {perm_2}")

            # cmd = f"{perm_1} && {perm_2}"
            # try:
            #     subprocess.run(cmd, shell=True, check=True)
            # except subprocess.CalledProcessError as e:
            #     print(f"Error: {e}")

            # Clone down the codeflare-sdk. We cannot get it from PyPi bc we make changes in the SDK to enable non-Openshift K8s support
            # clone_cf = f"git clone git@github.com:RohanSreerama5/codeflare-sdk.git"

            # logger.info(f"Running {clone_cf}")

            # cmd = f"{clone_cf}"
            # try:
            #     subprocess.run(cmd, shell=True, check=True)
            # except subprocess.CalledProcessError as e:
            #     print(f"Error: {e}")

            # Now we can deploy user Ray clusters

            # Import pieces from codeflare-sdk
            # from codeflare_sdk.cluster.cluster import Cluster, ClusterConfiguration
            # from codeflare_sdk.cluster.auth import KubeConfigFileAuthentication

            # kube_config_path = os.path.expanduser("~/.kube/config")
            # auth = KubeConfigFileAuthentication(
            #     kube_config_path = kube_config_path
            # )
            # auth.load_kube_config()

            # # Create and configure our cluster object (and appwrapper)
            # namespace = self.namespace
            # cluster_name = self.name
            # local_interactive = True

            # logger.info(f"local interactive: {local_interactive}, name: {cluster_name}, namespace: {namespace}, num_workers: {self.num_workers}, min_cpus: {self.cpus}")
            # logger.info(f"min_memory={self.memory}, num_gpus={self.num_gpus}, head_cpus={self.head_cpus}, head_memory={self.head_memory}")

            # cluster = Cluster(ClusterConfiguration(
            #     local_interactive=local_interactive,
            #     name=cluster_name,
            #     namespace=namespace,
            #     num_workers=self.num_workers,
            #     min_cpus=self.cpus,
            #     max_cpus=self.cpus,
            #     min_memory=self.memory,
            #     max_memory=self.memory,
            #     num_gpus=self.num_gpus,
            #     image="rayproject/ray:2.6.3", # needs to be set properly to ensure Ray version. currently Ray version is 2.7.0. needs to match client(local). This here is server's version
            #     instascale=False,
            #     head_cpus=self.head_cpus,
            #     head_memory=self.head_memory # don't need to surface head_node specs to user for now 
            #     # mcad=False 
            # ))

            cluster_name = self.name

            yaml_file_path = f'{cluster_name}.yaml'
            with open(yaml_file_path, 'r') as file:
                yaml_dict = yaml.safe_load(file)

            for item in yaml_dict['spec']['resources']['GenericItems'][:]:
                if 'generictemplate' in item and item['generictemplate']['apiVersion'] == 'route.openshift.io/v1':
                    yaml_dict['spec']['resources']['GenericItems'].remove(item)

            modified_yaml_str = yaml.dump(yaml_dict, default_flow_style=False)

            with open(yaml_file_path, 'w') as file:
                file.write(modified_yaml_str)

            # up the codeflare ray cluster 

            if not self.upped_cluster:
                self.upped_cluster = True
                self.cluster_obj.up()
                

            # DO NOT use ray client to send functions 

            # copy code to head node 
            # send function to head_node 

            self.cluster_obj.wait_ready(dashboard_check=False)

            # self.cluster_obj = cluster


            # STOP HERE


            # take in a context here instead here instead


        else:
            raise ValueError(f"Cluster provider {self.provider} not supported.")

        # self._update_from_sky_status()
        
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

            # sync_rh_cmd = "kubectl cp ../runhouse/ default/cpu-cluster-42-dc19-ray-head:runhouse"

            # consider modifying this so that the kubernetes namespace is parameterized 
            # also consider extracting the kubernetes pod name 

            # consider changing this to use kubectl get rayclusters instead 
            command = f"kubectl get pods -n {self.namespace} | grep {self.name}-head"
            pod_name = None
            try:
                
                output = subprocess.check_output(command, shell=True, text=True)
                
                lines = output.strip().split('\n')
                if lines:
                    pod_name = lines[0].split()[0]
                    print(f"Found matching pod: {pod_name}")
                else:
                    print("No matching pods found.")
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")

            kubectl_command = f"kubectl get pod {pod_name} -n default -o jsonpath='{{.status.podIP}}' | awk '{{print $1}}'"

            logger.info(f"Running {kubectl_command}")

            pod_ip = None

            try:
                result = subprocess.run(
                    kubectl_command,
                    shell=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if result.returncode == 0:
                    pod_ip = result.stdout.strip()
                    print(f"Pod IP Address: {pod_ip}")
                else:
                    print(f"Error: {result.stderr.strip()}")
            except Exception as e:
                print(f"Error: {e}")

            self.address = pod_ip

            # sync_rh_cmd = f"kubectl cp ../../../../runhouse/ default/{pod_name}:runhouse"
            sync_rh_cmd = f"kubectl cp ../runhouse/ default/{pod_name}:runhouse"
            

            cmd = f"{sync_rh_cmd}"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")

            # cur_name = f"this-16-worker-small-group-this-16-6z8zx"
            # sync_rh_cmd = f"kubectl cp ../runhouse/ default/this-16-worker-small-group-this-16-6z8zx:runhouse"
            

            # cmd = f"{sync_rh_cmd}"
            # try:
            #     subprocess.run(cmd, shell=True, check=True)
            # except subprocess.CalledProcessError as e:
            #     print(f"Error: {e}")

            # logger.info(f"Ran {cmd}")

            # copy runhouse to all worker nodes
            # from kubernetes import client, config

            # config.load_kube_config()
            # api_instance = client.CoreV1Api()
            # namespace = self.namespace
            # pods = api_instance.list_namespaced_pod(namespace)

            # for pod in pods.items:
            #     if f"{self.name}-worker" in pod.metadata.name:
            #         cp_cmd = f"kubectl cp ../runhouse/ default/{pod.metadata.name}:runhouse"
            #         try:
            #             subprocess.run(cp_cmd, shell=True, check=True)
            #             # print("Standard Output:")
            #             # print(result.stdout)
            #             # print("Standard Error:")
            #             # print(result.stderr)
            #         except subprocess.CalledProcessError as e:
            #             print(f"Error: {e}")

        install_rh_cmd = "pip install ./runhouse"

        cmd_1 = self.CLI_RESTART_CMD + (" --no-restart-ray" if not restart_ray else "")
        cmd_1 = f"{install_rh_cmd} && {cmd_1}"

        # status_codes = self.run(commands=[cmd_1])
        # if not status_codes[0][0] == 0:
        #     raise ValueError(f"Failed to restart server {self.name}.")
        # # As of 2023-15-May still seems we need this.
        # time.sleep(5)
        # return status_codes

        # better logging needed
        

        cmd = f"kubectl exec -n {self.namespace} {pod_name} -- {cmd_1}"

        logger.info(f"Running {cmd} on {self.name}")

        try:
            process = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            print(f"Command Output on {self.name}:")
            print(process.stdout)

        except subprocess.CalledProcessError as e:
            # Print the error message and standard error (stderr)
            print(f"Error: {e}")
            print(f"Command Error Output on {self.name}:")
            print(e.stderr)

        return 


    # def check_server(self, restart_server=True):
    #     return


    def up_if_not(self):
        """Bring up the cluster if it is not up. No-op if cluster is already up.

        Example:
            >>> rh.sagemaker_cluster("sagemaker-cluster").up_if_not()
        """
        if not self.is_up():
            self.up()
        return self

    def is_up(self) -> bool:
        """Check if the cluster is up.

        Example:
            >>> rh.sagemaker_cluster("sagemaker-cluster").is_up()
        """
        #cluster_status = self.cluster_obj.status()
        #return cluster_status[1]
        return True

    def teardown(self):
        """Teardown the SageMaker instance.

        Example:
            >>> rh.sagemaker_cluster(name="sagemaker-cluster").teardown()
        """
        self.cluster_obj.down()

    def status(self) -> dict:
        """
        Get status of SageMaker cluster.

        Example:
            >>> status = rh.sagemaker_cluster("sagemaker-cluster").status()
        """
        return self.cluster_obj.status()

    def ssh_tunnel():
        return

    def connect_server_client(self): 
        self.client = HTTPClient(host="127.0.0.1", port=DEFAULT_SERVER_PORT)
        return


    def _sync_runhouse_to_cluster(self, _install_url=None, env=None):
        return

    def _run_commands_with_ssh():
        return

    def _rsync(self, source: str, dest: str, up: bool, contents: bool = False):
        return

    @staticmethod
    def from_config(config: dict, dryrun=False):
        return KubernetesCluster(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        # config = super().config_for_rns

        config = {}

        # Also store the ssh keys for the cluster in RNS
        config.update(
            {
                "name": self.name,
                "num_workers": self.num_workers,
                "provider": self.provider,
                "memory": self.memory,
                "num_gpus": self.num_gpus,
                "head_cpus": self.head_cpus,
                "upped_cluster": self.upped_cluster,
                "resource_subtype": "KubernetesCluster"

            }
        )
        return config

    # def _get_sky_state(self):
    #     return

    


        
    
# mine = KubernetesCluster(
#             provider="kubernetes",
#             name="this-7",
#             namespace="default", # specify context, or namespace, actual EKS cluster, user. context or all three 
#             num_workers=1,
#             cpus=1,
#             memory=2,
#             num_gpus=0, # experimental: need to see how this really works 
#             head_cpus=1,
#             head_memory=4, # remove head_node, reg_node distinction 
#             # add arg for kube config 
#         )

# mine.up()

# context, path to kubecofnig, instance type and num nodes

# CPU:4 or GPU:4 (parse this) and pass to codeflare 
# support instance_type = [t3.micro]
# insatnce type = CPU:4 

# happy path: give name, context, instance type, num_instances 

# change naming later 
# instance_type = pod_type 
# num_instances = num_pods

# be able to regenerate cluster again if user breaks it 

# take code from head_node (on notebook) and put it on a orchestration node (pipeline) for prod
#  create new ray cluster, takes code out of Github, and puts it on head node and runs it (could use docker)