#!/bin/bash

echo "===== Install Kubectl ====="

curl -LO https://storage.googleapis.com/kubernetes-release/release/v1.14.8/bin/darwin/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl

echo "===== Install KFctl ====="

wget https://github.com/kubeflow/kfctl/releases/download/v1.0-rc.1/kfctl_v1.0-rc.1-0-g963c787_darwin.tar.gz -t ./
tar -xvf *.tar.gz

PATH=$PATH:$PWD