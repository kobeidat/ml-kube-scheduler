from kubernetes import config, client, utils
from prometheus import query_prometheus_cpu
import numpy as np
from time import sleep


def cpu_variance():
    response = query_prometheus_cpu(5)
    cpu_usage = []

    for node_data in response:
        cpu_usage.append(float(node_data["value"][1]))

    cpu_usage = np.array(cpu_usage) * 100
    var = np.var(cpu_usage)
    print(cpu_usage)
    print(var)

def evaluate():
    print("Before deployment:")
    cpu_variance()
    utils.create_from_yaml(client, "pods/cpu-stress-deploy.yaml")
    print("After deployment:")
    cpu_variance()

    sleep(60)
    print("After 60 seconds:")
    cpu_variance()


config.load_kube_config()
client = client.ApiClient()
