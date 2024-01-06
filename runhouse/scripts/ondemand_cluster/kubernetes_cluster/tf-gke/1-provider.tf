# https://registry.terraform.io/providers/hashicorp/google/latest/docs
provider "google" {
  project = "runhouse-prod" # project name needs to exist in GCP already
  region  = "us-east1"
}

# https://www.terraform.io/language/settings/backends/gcs
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}
