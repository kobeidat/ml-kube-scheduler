from kubernetes import config, client, utils
from prometheus import query_prometheus_cpu, get_timestamp
import numpy as np
from time import sleep
import json
import pandas as pd
import sys

# ======================================================= #
# do zrobienia:
# wiecej metryk do ewaluacji (np wariancja RAM i inne)
# automatyzacja ewaluacji
# zapisywanie wynikow do pliku
# tworzenie wykresow
# ======================================================= #

EVALS = 4 # number of evaluation points
INTERVAL = 1 # seconds

def cpu_variance():
    response = query_prometheus_cpu(1)
    cpu_usage = []

    for node_data in response:
        cpu_usage.append(float(node_data["value"][1]))

    cpu_usage = np.array(cpu_usage) * 100
    var = np.var(cpu_usage)
    return var

def evaluate(deployment, metric):
    metrics = {
        "cpu_var": cpu_variance,
        "test": np.random.uniform
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")

    # config.load_kube_config()
    # cli = client.ApiClient()

    values = []
    values.append(metrics[metric]())
    print("Before deployment:")
    print(f"CPU variance: {np.round(values[-1], 4)}")

    # utils.create_from_yaml(cli, deployment)
    values.append(metrics[metric]())
    print("After deployment:")
    print(f"CPU variance: {np.round(values[-1], 4)}")

    for i in range(EVALS):
        sleep(INTERVAL)
        values.append(metrics[metric]())
        print(f"After {(i + 1) * INTERVAL} seconds:")
        print(f"CPU variance: {np.round(values[-1], 4)}")

    file = open("logs/eval.json", "a", encoding="utf-8")
    data = {"timestamp": get_timestamp(), "scheduler": "default"}

    for i in range(EVALS + 2):
        data[f"val{i}"] = values[i]

    file.write(json.dumps(data, ensure_ascii=False) + "\n")
    file.close()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise ValueError(f"Not enough arguments")

    deployment = sys.argv[1]
    if len(sys.argv) > 2:
        metric = sys.argv[2]
    else:
        metric = "test"
    
    evaluate(deployment, metric)
    df = pd.read_json("logs/eval.json", lines=True)
    print(df)
