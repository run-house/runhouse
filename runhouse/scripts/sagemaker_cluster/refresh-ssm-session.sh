#!/bin/bash

# Note: Adapted from: https://github.com/aws-samples/sagemaker-ssh-helper/blob/main/sagemaker_ssh_helper/sm-connect-ssh-proxy
# This skips creating SSH keys and storing them in S3, which happened when the cluster was initially upped
# and does not need to be repeated when reconnecting with the cluster.

INSTANCE_ID="$1"
SSH_KEY="$2"
CURRENT_REGION="$3"
shift 3
PORT_FWD_ARGS=$*

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
