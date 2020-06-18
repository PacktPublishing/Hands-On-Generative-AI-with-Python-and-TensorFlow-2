#!/bin/bash

export VERSION_TAG=$(date +%s)
export REGISTRY_NAME="kubeflowcontainers"
export RESOURCE_GROUP_NAME="KubeTest"
export NAME="KubeTestCluster"
export REGISTRY_PATH=${REGISTRY_NAME}".azurecr.io"
export TRAIN_IMG_PATH=${REGISTRY_PATH}/mnist-train:${VERSION_TAG}
export WORKING_DIR=${PWD}/examples/mnist



echo "===== login to Azure Container Registry ====="

az acr login --name ${REGISTRY_NAME}

echo "===== clone Kubeflow examples ====="

git clone https://github.com/kubeflow/examples.git

echo "===== build Docker Container for MNIST example ====="

docker build $WORKING_DIR -t $TRAIN_IMG_PATH -f $WORKING_DIR/Dockerfile.model
docker push $TRAIN_IMG_PATH

echo "===== Create Bucket for Azure Storage ======"

export AZURE_STORAGE_ACCOUNT="kubeflowstorage"
export AZURE_STORAGE_KEY="4FaoWf5sBGDgIHRmTU48seFstrK1aJmJio8jxIfi5p4W7opJv2G6KsY0ITlONL3SQAq3+4xFFlQmZ8j/BFJLCg=="
export BUCKET_NAME="mnist-training"


export BUCKET_PATH=${BUCKET_NAME}/${VERSION_TAG}
kustomize edit add configmap mnist-map-training   --from-literal=modelDir=https://kubeflowstorage.blob.core.windows.net/${BUCKET_PATH}/
kustomize edit add configmap mnist-map-training   --from-literal=exportDir=https://kubeflowstorage.blob.core.windows.net/${BUCKET_PATH}/export


https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv2.0.3