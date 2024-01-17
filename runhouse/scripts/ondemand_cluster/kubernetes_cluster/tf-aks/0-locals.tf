locals {
  env                 = "dev"
  region              = "eastus2"
  resource_group_name = "skyakstestrg"
  eks_name            = "skyakstest"  # Note: AKS cluster name will be dev-{eks_name}
  eks_version         = "1.28"
}
