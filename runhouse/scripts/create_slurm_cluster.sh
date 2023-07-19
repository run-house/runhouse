#!/bin/bash

echo "Starting install commands for Slurm...."

# Name to assign for this node
node_name="node01"

# Number of CPUs available on the node
cpu_count=$(nproc)
echo "CPUs on this node: $cpu_count"

sudo apt install net-tools

# Get the private IP address of the cluster using ifconfig
private_ip=$(ifconfig | grep -oP '(?<=inet )[\d.]+(?=  netmask)' | awk 'NR==2 {print}')
echo "Private IP Address: $private_ip"

# Write the node name to the /etc/hostname file with elevated privileges
echo "$node_name" | sudo tee /etc/hostname >/dev/null

# Write the private IP address and node name to /etc/hosts with elevated privileges
echo "$private_ip $node_name" | sudo tee /etc/hosts >/dev/null

# Install Slurm using apt-get
sudo apt-get install -y slurm-wlm

sudo mkdir -p /etc/slurm-llnl
cd /etc/slurm-llnl

sudo cp /usr/share/doc/slurm-client/examples/slurm.conf.simple.gz .

sudo gzip -d slurm.conf.simple.gz

sudo mv slurm.conf.simple slurm.conf

# Update a couple of the fields in the slurm.conf
sudo sed -i 's/PartitionName=debug Nodes=server Default=YES MaxTime=INFINITE State=UP/PartitionName=debug Nodes=node[01] Default=YES MaxTime=INFINITE State=UP/' slurm.conf
sudo sed -i "s/SlurmctldHost=workstation/SlurmctldHost=$node_name/" slurm.conf
sudo sed -i "s|SlurmdPidFile=/run/slurmd.pid|SlurmdPidFile=/var/run/slurmd.pid|" slurm.conf
sudo sed -i "s|SlurmctldPidFile=/run/slurmctld.pid|SlurmctldPidFile=/var/run/slurmctld.pid|" slurm.conf
sudo sed -i "s/NodeName=server CPUs=1 State=UNKNOWN/NodeName=$node_name NodeAddr=$private_ip CPUs=1 State=UNKNOWN/" slurm.conf


# Create the cgroup.conf file and add the specified contents:q!
sudo touch cgroup.conf
sudo bash -c 'cat << EOF > cgroup.conf
CgroupMountpoint="/sys/fs/cgroup"
CgroupAutomount=yes
CgroupReleaseAgentDir="/etc/slurm-llnl/cgroup"
AllowedDevicesFile="/etc/slurm-llnl/cgroup_allowed_devices_file.conf"
ConstrainCores=no
TaskAffinity=no
ConstrainRAMSpace=yes
ConstrainSwapSpace=no
ConstrainDevices=no
AllowedRamSpace=100
AllowedSwapSpace=0
MaxRAMPercent=100
MaxSwapPercent=100
MinRAMSpace=30
EOF
'

# Create the cgroup.conf file and add the specified contents
sudo touch cgroup_allowed_devices_file.conf
sudo bash -c 'cat << EOF > cgroup_allowed_devices_file.conf
/dev/null
/dev/urandom
/dev/zero
/dev/sda*
/dev/cpu/*/*
/dev/pts/*
/shared*
EOF
'

sudo systemctl enable munge
sudo systemctl start munge
sudo systemctl enable slurmd
sudo systemctl start slurmd
sudo systemctl enable slurmctld
sudo systemctl start slurmctld

echo "Running sinfo command...."
sinfo

# Check if a file exists
if sinfo | grep -q "idle"; then
    echo "Slurm properly configured :)"
else
    echo "Failed to configure Slurm :("
fi
