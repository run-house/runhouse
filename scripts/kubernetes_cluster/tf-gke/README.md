Basic GKE cluster setup guide in Terraform

To spin up a GKE cluster in GCP, please follow the below steps:

`brew install --cask google-cloud-sdk`

Ensure your have GCP access first with the appropriate permissions level.

`gcloud auth application-default login` This will prompt you to login via browser, using your gmail account.

`gcloud auth application-default set-quota-project runhouse-prod`

`terraform init`

`terraform validate`

`terraform plan -out gke_plan`

`terraform apply "gke_plan"`


`gcloud config set project runhouse-prod`

`gcloud components install gke-gcloud-auth-plugin` This is neccesary for kubectl with GKE to work

Finally, go to your GKE cluster in the GCP console and copy the command found by pressing the `Connect` tab. Run this command.

Test your access to the GKE cluster by running `kubectl get nodes`
