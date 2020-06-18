#!/bin/bash

export AZ_USERNAME="joseph.j.babcock@gmail.com"
export AZ_PASSWD="TapFav53**"
export RESOURCE_GROUP_NAME="KubeTest"
export LOCATION="eastus"
export NAME="KubeTestCluster"
export AGENT_SIZE="Standard_ND6"
export AGENT_COUNT="3"
export KF_NAME="AZKFTest"
export BASE_DIR=/opt
export KF_DIR=${BASE_DIR}/${KF_NAME}
export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v1.0-branch/kfdef/kfctl_k8s_istio.yaml"

echo "===== Install Azure CLI ====="

brew install azure-cli

echo "===== Login to Azure ====="

az login -u ${AZ_USERNAME} -p ${AZ_PASSWD}

echo "===== Create Resource Group ====="

az group create -n ${RESOURCE_GROUP_NAME} -l ${LOCATION}

echo "===== Create AKS Cluster in Resource Group ====="

az aks create -g ${RESOURCE_GROUP_NAME} -n ${NAME} -s ${AGENT_SIZE} -c ${AGENT_COUNT} -l ${LOCATION} --generate-ssh-keys

echo "===== Get Azure Credentials ====="

az aks get-credentials --resource-group ${RESOURCE_GROUP_NAME} --name ${NAME}

echo "===== Install Kubeflow on AKS ====="

if [ -d ${KF_DIR} ]; then rm -r -f ${KF_DIR}; fi
mkdir -p ${KF_DIR}
cd ${KF_DIR}
kfctl apply -V -f ${CONFIG_URI}






