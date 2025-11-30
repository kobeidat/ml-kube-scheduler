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
```
```
helm repo update
helm install prometheus prometheus-community/prometheus --set server.persistentVolume.enabled=false
```
Apply test pod
```
kubectl apply -f scheduler-deployment.yaml
kubectl apply -f pods/cpu-stress.yaml
```
## Evaluation
Compare custom Kubernetes scheduler with the default one.
```
python evaluation.py -pods <path> [<path> ...] [-scheduler <name>] [-metric <name>] [-graph]
```
Allowed metrics:
* `cpu_var` - CPU variance
* `mem_var` - memory variance
* `test` - metric for testing, random numbers, default

Allowed schedulers:
* `ml` - our custom scheduler, default
### Useful commands
```
kubectl get pods [-n kube-system]
kubectl logs -n kube-system my-scheduler-<id>
kubectl port-forward svc/prometheus-server 9090:80
```
## TODO
- [x] motywacje do rozwiązania jako początek pierwszego rozdziału
- [x] opis problemu pierwszy rozdział szczegółowo
- [x] user stories do wymagań w tabeli
- [x] MoSCoW w tabeli obok user stories
- [ ] wykorzystane technologie w 1 rozdziale
- [ ] aktorzy do 1 rozdziału
- [ ] dalsza część po wstępie bardziej szczegółowa
- [ ] do każdego punktu wstęp
- [ ] 4-5 linijek na każdy punkt
- [ ] !!! rysunki, schematy, diagramy, screeny, wygląd architektury powiązań, przepływu informacji !!!
- [ ] opis do algorytmów
- [ ] planowane poprawki mogą zostać zmienione na wykonane poprawki
