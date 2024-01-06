To spin up an EKS cluster in AWS, 

Simply change the commented fields in main.tf and run the standard TF commands. Ensure that you have the AWS CLI setup with the correct permissions and access keys, etc. 

`terraform init` 

`terraform validate` 

`terraform plan -out eks_plan` 

`terraform apply "eks_plan"` 

You should also run `aws eks update-kubeconfig --region us-east-1 --name NAME_OF_EKS_CLUSTER` to update your kubeconfig. 
