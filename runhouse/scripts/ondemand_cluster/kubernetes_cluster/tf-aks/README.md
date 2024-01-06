To spin up an AKS cluster in Azure, please follow the below steps. 

Clone down this repository and modify locals.tf to contain your desired region, resource group name, and cluster name. Optionally modify the K8s version to reflect the 
latest one. 

In subnets.tf, you can use your existing subnets or create new ones. If you do not modify it, it will create new subnets in Azure. 

Lastly, you can also modify some settings in aks.tf to reflect your need. The area of most interest is the `default node pool` where you may adjust the VM type, auto scaling,
and min / max node count. 

Once you are ready with your TF scripts, you will begin by logging into Azure via the CLI. 

Open a terminal and run `brew install azure-cli`. Then, run `az login`. 
NOTE: You may need to add a `TENANT_ID` argument to the `az login` command. 

Authenticate with `az login`, making sure your terminal has access to your Azure Cloud account. 

Next, find your subscription ID by running `az account list`. Copy your subscription's ID and then run 

`az account set --subscription SUBSCRIPTION_ID` 

Finally, run your standard TF commands to deploy this AKS cluster.

`terraform init` 

`terraform validate` 

`terraform plan -out tf_plan` 

`terraform apply "tf_plan"`

To get access to your AKS cluster, you will need its kubeconfig locally. To obtain this run, 

`az aks get-credentials --resource-group RESOURCE_GROUP_NAME --name AKS_CLUSTER_NAME`. 

Note now that ~/.kube/config's contents will be updated with the kubeconfig of your AKS cluster.

Finally, test your connection by running `kubectl get nodes` 




