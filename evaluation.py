from kubernetes import config, client, utils
from prometheus import query_prometheus_cpu, get_timestamp
import numpy as np
from time import sleep
import json
import pandas as pd

# ================================================== #
# do zrobienia:
# wiecej metryk do ewaluacji (np wariancja RAM i inne)
# automatyzacja ewaluacji
# zapisywanie wynikow do pliku
# tworzenie wykresow
# ================================================== #

EVALS = 4

def cpu_variance():
    response = query_prometheus_cpu(1)
    cpu_usage = []

    for node_data in response:
        cpu_usage.append(float(node_data["value"][1]))

    cpu_usage = np.array(cpu_usage) * 100
    var = np.var(cpu_usage)
    return var

def evaluate():
    config.load_kube_config()
    client = client.ApiClient()

    vars = []
    vars.append(cpu_variance())
    print("Before deployment:")
    print(f"CPU variance: {np.round(vars[-1], 4)}")

    utils.create_from_yaml(client, "pods/cpu-stress-deploy.yaml")
    vars.append(cpu_variance())
    print("After deployment:")
    print(f"CPU variance: {np.round(vars[-1], 4)}")

    for i in range(EVALS):
        sleep(1)
        vars.append(cpu_variance())
        print(f"After {(i + 1) * 30} seconds:")
        print(f"CPU variance: {np.round(vars[-1], 4)}")

    file = open("logs/eval.json", "a", encoding="utf-8")
    data = {"timestamp": get_timestamp()}

    for i in range(EVALS + 2):
        data[f"var{i}"] = vars[i]

    file.write(json.dumps(data, ensure_ascii=False) + "\n")
    file.close()


evaluate()
df = pd.read_json("logs/eval.json", lines=True)
print(df)
