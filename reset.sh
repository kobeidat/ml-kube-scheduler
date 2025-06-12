#!/bin/bash

set -euo pipefail

echo "🔥 STOPPING Docker and Minikube..."
sudo systemctl stop docker || true
minikube stop || true

echo "🧼 Cleaning up Docker..."
docker rm -f $(docker ps -aq) 2>/dev/null || true
docker rmi -f $(docker images -q) 2>/dev/null || true
docker volume rm $(docker volume ls -q) 2>/dev/null || true
docker network rm $(docker network ls | grep -v "bridge\|host\|none" | awk '{print $1}') 2>/dev/null || true

echo "🗑️ Deleting Docker system data..."
sudo rm -rf /var/lib/docker /var/lib/containerd /etc/docker
rm -rf ~/.docker

echo "🧹 Wiping Minikube..."
minikube delete --all --purge || true
rm -rf ~/.minikube
rm -rf ~/.kube

echo "🪓 Nuking Helm..."
helm uninstall $(helm list --all-namespaces -q) --all-namespaces 2>/dev/null || true
rm -rf ~/.helm
rm -rf ~/.config/helm
rm -rf ~/.cache/helm

echo "❌ Optionally removing binaries (comment out if you want to keep them)..."
sudo rm -f /usr/local/bin/docker
sudo rm -f /usr/local/bin/containerd
sudo rm -f /usr/local/bin/minikube
sudo rm -f /usr/local/bin/helm

echo "✅ DONE: Docker, Minikube, and Helm have been nuked from orbit."
