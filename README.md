How to run

```shell
minikube start

eval $(minikube -p minikube docker-env)
docker build -t custom-scheduler:latest ./scheduler

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/prometheus -f manifests/prometheus-values.yaml

kubectl apply -f manifests/custom-scheduler-rbac.yaml
kubectl apply -f manifests/custom-scheduler-deployment.yaml
```
