#!/bin/bash

# Adapted from: https://github.com/aws-samples/sagemaker-ssh-helper/blob/main/sagemaker_ssh_helper/sm-connect-ssh-proxy
# Creates an SSM session and sets up port forwarding to the cluster through localhost for an SSH port and HTTP port
# Optionally creates new SSH keys if they do not already exist

set -e

INSTANCE_ID="$1"
SSH_AUTHORIZED_KEYS="$2"
SSH_KEY="$3"
CURRENT_REGION="$4"
shift 4
PORT_FWD_ARGS=$*

echo "INSTANCE_ID: $INSTANCE_ID"
echo "SSH_AUTHORIZED_KEYS: $SSH_AUTHORIZED_KEYS"
echo "CURRENT_REGION: $CURRENT_REGION"
echo "PORT_FWD_ARGS: $PORT_FWD_ARGS"

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

echo "Running SSM commands at region ${CURRENT_REGION} to copy public key to ${INSTANCE_ID}"

# Copy the public key from the s3 bucket to the authorized_keys.d directory on the cluster
cp_command="aws s3 cp --recursive \"${SSH_AUTHORIZED_KEYS}\" /root/.ssh/authorized_keys.d/"

# Copy the SSH public key onto the cluster to the root directory, then copy from the root to /etc
send_command=$(aws ssm send-command \
    --region "${CURRENT_REGION}" \
    --instance-ids "${INSTANCE_ID}" \
    --document-name "AWS-RunShellScript" \
    --comment "Copy public key for SSH helper" \
    --timeout-seconds 30 \
    --parameters "commands=[
        'mkdir -p /root/.ssh/authorized_keys.d/',
        '$cp_command',
        'ls -la /root/.ssh/authorized_keys.d/',
        'cat /root/.ssh/authorized_keys.d/* > /root/.ssh/authorized_keys',
        'cat /root/.ssh/authorized_keys'
      ]" \
    --no-cli-pager --no-paginate \
    --output json)

json_value_regexp='s/^[^"]*".*": \"\(.*\)\"[^"]*/\1/'

cp_command="cp -r /root/.ssh/authorized_keys.d/* /etc/ssh/authorized_keys.d"
echo "Copying keys from root folder to etc folder: $cp_command"

send_command=$(aws ssm send-command \
    --region "${CURRENT_REGION}" \
    --instance-ids "${INSTANCE_ID}" \
    --document-name "AWS-RunShellScript" \
    --comment "Copy public key to /etc/ssh folder on cluster" \
    --timeout-seconds 30 \
    --parameters "commands=[
        'mkdir -p /etc/ssh/authorized_keys.d/',
        '$cp_command',
        'ls -la /etc/ssh/authorized_keys.d/',
        'cat /etc/ssh/authorized_keys.d/* > /etc/ssh/authorized_keys',
        'ls -la /etc/ssh/authorized_keys'
      ]" \
    --no-cli-pager --no-paginate \
    --output json)

json_value_regexp='s/^[^"]*".*": \"\(.*\)\"[^"]*/\1/'


send_command=$(echo "$send_command" | python -m json.tool)
command_id=$(echo "$send_command" | grep "CommandId" | sed -e "$json_value_regexp")
echo "Got command ID: $command_id"

# Wait a little bit to prevent strange InvocationDoesNotExist error
sleep 5

for i in $(seq 1 15); do
  # Switch to unicode for AWS CLI to properly parse output
  export LC_CTYPE=en_US.UTF-8
  command_output=$(aws ssm get-command-invocation \
      --instance-id "${INSTANCE_ID}" \
      --command-id "${command_id}" \
      --no-cli-pager --no-paginate \
      --output json)
  command_output=$(echo "$command_output" | python -m json.tool)
  command_status=$(echo "$command_output" | grep '"Status":' | sed -e "$json_value_regexp")
  output_content=$(echo "$command_output" | grep '"StandardOutputContent":' | sed -e "$json_value_regexp")
  error_content=$(echo "$command_output" | grep '"StandardErrorContent":' | sed -e "$json_value_regexp")

  echo "Command status: $command_status"
  if [[ "$command_status" != "Pending" && "$command_status" != "InProgress" ]]; then
    echo "Command output: $output_content"
    if [[ "$error_content" != "" ]]; then
      echo "Command error: $error_content"
    fi
    break
  fi
  sleep 1
done

if [[ "$command_status" != "Success" ]]; then
  echo "Error: Command didn't finish successfully in time"
  exit 2
fi

echo "Connecting to $INSTANCE_ID as proxy and starting port forwarding with the args: $PORT_FWD_ARGS"

# We don't use AWS-StartPortForwardingSession feature of SSM here, because we need port forwarding in both directions
#  with -L and -R parameters of SSH. This is useful for forwarding the PyCharm license server, which needs -R option.
#  SSM allows only forwarding of ports from the server (equivalent to the -L option).
# shellcheck disable=SC2086
proxy_command="aws ssm start-session\
 --reason 'Local user started SageMaker SSH Helper'\
 --region '${CURRENT_REGION}'\
 --target '${INSTANCE_ID}'\
 --document-name AWS-StartSSHSession\
 --parameters portNumber=%p"

# shellcheck disable=SC2086
ssh -4 -T -o User=root -o IdentityFile="${SSH_KEY}" -o IdentitiesOnly=yes \
  -o ProxyCommand="$proxy_command" \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -o PasswordAuthentication=no \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  $PORT_FWD_ARGS "$INSTANCE_ID"
