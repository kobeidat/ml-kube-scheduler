# Kubernetes scheduler with ML
A customizable Kubernetes scheduler with usage of machine learning.
## How to run
Prepare Kubernetes environment
```
minikube start --nodes=3 --cpus=2 --memory=2048 --driver=docker
docker build -t my-scheduler:latest .
minikube image load my-scheduler:latest
```
if repository not added
```
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/prometheus --set server.persistentVolume.enabled=false
```
Apply test pod
```
kubectl apply -f pods/cpu-stress-deployment.yaml
```
## Evaluation
```
python evaluation.py -deployment <path> [-scheduler <name>] [-metric <name>]
```
Allowed metrics:
* `cpu_var` - CPU variance
* `test` - metric for testing, random numbers, default

Allowed schedulers:
* `ml` - our custom scheduler, default
### Useful commands
```
kubectl get pods -n kube-system
kubectl logs -n kube-system <scheduler-name>
kubectl port-forward svc/prometheus-server 9090:80
```
