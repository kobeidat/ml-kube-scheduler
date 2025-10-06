# Kubernetes scheduler with ML
## How to run
Prepare Kubernetes environment
```
minikube start --nodes=3 --cpus=2 --memory=2048 --driver=docker
docker build -t my-scheduler:latest .
minikube image load my-scheduler:latest
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/prometheus
kubectl apply -f rbac.yaml
kubectl apply -f deployment.yaml
```
Apply test pod
```
kubectl apply -f test-pod.yaml
```
## Evaluation
```
kubectl get pods -n kube-system
kubectl logs -n kube-system <scheduler-name>
```
