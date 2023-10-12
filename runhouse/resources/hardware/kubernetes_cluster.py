from pathlib import Path
import subprocess
import logging
import time
import os
import yaml

from runhouse.resources.hardware.cluster import Cluster as cluser
from runhouse.servers.http import DEFAULT_SERVER_PORT, HTTPClient

logger = logging.getLogger(__name__)

class KubernetesCluster(cluser):

    def __init__(
            self, 
            name, 
            instance_type: str = None, 
            num_instances: int = None, 
            provider: str = "kubernetes", 
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
            cluster_obj = None,
            pod_name: str = None, # name of K8s pod head node is on 
            kube_config_path: str = None,
            cluster_setup_complete=False,
            **kwargs
    ):
        
        self.instance_type = instance_type
        self.namespace = namespace
        self.num_workers = num_workers
        self.num_gpus = 0       # causes issues if you do not set this to an integer (codeflare related)
        self.kube_config_path = kube_config_path
        self.pod_name = pod_name
        self.cluster_setup_complete = cluster_setup_complete
       
        self.cpus = cpus
        self.memory = memory
        
        self.head_cpus = head_cpus
        self.head_memory = head_memory
        self.provider = provider

        if self.instance_type is not None and ":" in self.instance_type:
            parts = self.instance_type.split(":")
            compute_type = None
            num_cores = None
            if len(parts) == 2:
                compute_type = parts[0].strip()
                num_cores = int(parts[1].strip())

            
            if "CPU" in compute_type:
                self.cpus = num_cores
                if self.head_cpus is None: # user didn't explicitly set head_cpus 
                    self.head_cpus = num_cores
            elif "GPU" in compute_type: # experimental
                self.num_gpus = num_cores


        if self.head_memory is None: # user didn't explicitly set head_memory 
            self.head_memory = self.memory

        # If user does not specify head_cpus and head_memory we set it equal to cpus and memory

        

        super().__init__(name=name, ips=[self.namespace], ssh_creds={}, dryrun=dryrun)

        # Setup codeflare-sdk on local
        if not self.on_this_cluster(): # ensure this actually works

            from codeflare_sdk.cluster.cluster import Cluster, ClusterConfiguration
            from codeflare_sdk.cluster.auth import KubeConfigFileAuthentication

            kube_config_path = None
            if self.kube_config_path is not None:
                kube_config_path = os.path.expanduser(self.kube_config_path)
            else:
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

    def is_up(self) -> bool:
        """Check if the cluster is up.

        Example:
            >>> rh.kubernetes_cluster("kubernetes-cluster").is_up()
        """
        cluster_status = self.cluster_obj.status()
        return cluster_status[1]

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

        # setup EKS K8s cluster for codeflare sdk usage

        if not self.cluster_setup_complete:

            # Install Kuberay Operator (Ensure helm is installed locally)
            add_kuberay = f"helm repo add kuberay https://ray-project.github.io/kuberay-helm/"
            logger.info(f"Running {add_kuberay}")
            cmd = f"{add_kuberay}"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")

            install_kuberay = f"helm install kuberay-operator kuberay/kuberay-operator"
            logger.info(f"Running {install_kuberay}")
            cmd = f"{install_kuberay}"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")

            # # Install Codeflare Operator (Ensure gnu sed is installed locally)
            clone_cf_operator = f"git clone git@github.com:RohanSreerama5/codeflare-operator.git"
            cd_cmd = f"cd codeflare-operator"
            make_install = f"make install -e SED=/opt/homebrew/opt/gnu-sed/libexec/gnubin/sed"
            make_deploy = f"make deploy -e SED=/opt/homebrew/opt/gnu-sed/libexec/gnubin/sed"

            logger.info(f"Running {clone_cf_operator} && {cd_cmd} && {make_install} && {make_deploy}")

            cmd = f"{clone_cf_operator} && {cd_cmd} && {make_install} && {make_deploy}"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")

            # Setup permissions so Codeflare Operator can manage Ray clusters (Ensure kubectl is setup locally)
            perm_1 = f"kubectl apply -f mcad-controller-ray-clusterrole.yaml"
            perm_2 = f"kubectl apply -f mcad-controller-ray-clusterrolebinding.yaml"

            logger.info(f"Running {perm_1} && {perm_2}")

            cmd = f"{perm_1} && {perm_2}"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")

            self.cluster_setup_complete = True

        # Clone down the codeflare-sdk. We cannot get it from PyPi bc we make changes in the SDK to enable non-Openshift K8s support
        # clone_cf = f"git clone git@github.com:RohanSreerama5/codeflare-sdk.git"

        # logger.info(f"Running {clone_cf}")

        # cmd = f"{clone_cf}"
        # try:
        #     subprocess.run(cmd, shell=True, check=True)
        # except subprocess.CalledProcessError as e:
        #     print(f"Error: {e}")

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
        self.cluster_obj.up()

        self.cluster_obj.wait_ready(dashboard_check=False)

        # take in a context here instead here instead

        
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

            # Get pod name
            command = f"kubectl get pods -n {self.namespace} | grep {self.name}-head"
            pod_name = None

            logger.info("Obtaining name of K8s pod that the head node is running on")
            logger.info(f"Running {command}")

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


            self.pod_name = pod_name


            # Get pod's IP address
            kubectl_command = f"kubectl get pod {pod_name} -n default -o jsonpath='{{.status.podIP}}' | awk '{{print $1}}'"

            logger.info("Obtaining IP address of K8s pod that your head node is running on")
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

            # Set self.address to pod's IP address (this may not be neccesary to do)
            self.address = pod_ip

            # Sync runhouse code over to pod
            sync_rh_cmd = f"kubectl cp ../runhouse/ default/{pod_name}:runhouse"

            logger.info("Syncing runhouse code to cluster")
            logger.info(f"Running: {sync_rh_cmd}")
            
            cmd = f"{sync_rh_cmd}"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")

            

        install_rh_cmd = "pip install ./runhouse"

        restart_runhouse = self.CLI_RESTART_CMD + (" --no-restart-ray" if not restart_ray else "")


        # Install runhouse via pip 
        cmd = f"kubectl exec -n {self.namespace} {pod_name} -- {install_rh_cmd}"

        logger.info("Installing runhouse python dependencies on the cluster")
        logger.info(f"Running {cmd} on {self.name}")

        try:
            process = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            print(f"Command Output on {self.name}:")
            print(process.stdout)

        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print(f"Command Error Output on {self.name}:")
            print(e.stderr)

        # Restart runhouse HTTP server
        cmd = f"kubectl exec -n {self.namespace} {pod_name} -- {restart_runhouse}"

        logger.info("Restarting the HTTP server on the cluster")
        logger.info(f"Running {cmd} on {self.name}")

        try:
            process = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            print(f"Command Output on {self.name}:")
            print(process.stdout)

        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print(f"Command Error Output on {self.name}:")
            print(e.stderr)


        # Set up port-forward from local to pod port 50052
        command = f"kubectl port-forward {pod_name} 50052:{DEFAULT_SERVER_PORT}"

        logger.info("Setting up port-forward")
        logger.info(f"Running {command}")

        try:
            # Start the command in the background
            process = subprocess.Popen(command, shell=True, text=True)

        except subprocess.CalledProcessError as e:
            # Check if the error message contains "Address already in use"
            if "Address already in use" in str(e):
                print("Port 50052 is already in use. Please free up the port.")
            else:
                print(f"An error occurred while running the command: {e}")


        return 


    def up_if_not(self):
        """Bring up the cluster if it is not up. No-op if cluster is already up.

        Example:
            >>> rh.kubernetes_cluster("kubernetes-cluster").up_if_not()
        """
        if not self.is_up():
            self.up()
        return self


    def teardown(self):
        """Teardown the Kubernetes instance.

        Example:
            >>> rh.kubernetes_cluster(name="kubernetes-cluster").teardown()
        """
        self.cluster_obj.down()
        self.address = None

    def status(self) -> dict:
        """
        Get status of Kubernetes cluster.

        Example:
            >>> status = rh.kubernetes_cluster("kubernetes-cluster").status()
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
        config = super().config_for_rns

        config.update(
            {
                "num_workers": self.num_workers,
                "provider": self.provider,
                "cpus": self.cpus,
                "memory": self.memory,
                "num_gpus": self.num_gpus,
                "head_cpus": self.head_cpus,
                "head_memory": self.head_memory,
                "namespace": self.namespace,
                "pod_name": self.pod_name,
                "kube_config_path": self.kube_config_path

            }
        )
        return config

    def shell(self): # experimental: requires testing 
        cmd = f"kubectl exec -it {self.pod_name} -- /bin/bash"
        try:
            process = subprocess.run(cmd, shell=True, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")


    


        
    
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