# Kubernetes Scheduler with ML

## How to run
```
docker build -t my-scheduler:latest .
kubectl apply -f rbac.yaml
kubectl apply -f deployment.yaml
kubectl apply -f test-pod.yaml
```
