#!/bin/bash

# Note: Adapted from: https://github.com/aws-samples/sagemaker-ssh-helper/blob/main/sagemaker_ssh_helper/sm-connect-ssh-proxy
# This skips creating SSH keys and adding the public key to the authorized keys list in S3, which happened when the
# cluster was initially upped and does not need to be repeated when reconnecting with the cluster.

INSTANCE_ID="$1"
SSH_KEY="$2"
CURRENT_REGION="$3"
shift 3
PORT_FWD_ARGS=$*

instance_status=$(aws ssm describe-instance-information --filters Key=InstanceIds,Values="$INSTANCE_ID" --query 'InstanceInformationList[0].PingStatus' --output text)

echo "Cluster status: $instance_status"

if [[ "$instance_status" != "Online" ]]; then
  echo "Error: Cluster is offline."
  exit 1
fi

AWS_CLI_VERSION=$(aws --version)

# Check if the AWS CLI version contains "aws-cli/2."
if [[ $AWS_CLI_VERSION == *"aws-cli/2."* ]]; then
  echo "AWS CLI version: $AWS_CLI_VERSION"
else
  echo "Error: AWS CLI version must be v2. Please update your AWS CLI version."
  exit 1
fi

echo "Starting SSH over SSM proxy"

proxy_command="aws ssm start-session\
 --reason 'Local user started SageMaker SSH Helper'\
 --region '${CURRENT_REGION}'\
 --target '${INSTANCE_ID}'\
 --document-name AWS-StartSSHSession\
 --parameters portNumber=22"

ssh -4 -T -o User=root -o IdentityFile="${SSH_KEY}" -o IdentitiesOnly=yes \
  -o ProxyCommand="$proxy_command" \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -o PasswordAuthentication=no \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  $PORT_FWD_ARGS "$INSTANCE_ID"
