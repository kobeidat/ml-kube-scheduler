from kubernetes import config, client, utils
from prometheus import query_prometheus_cpu, get_timestamp
import numpy as np
import pandas as pd
from time import sleep
import json
import sys
import yaml
import argparse


# ======================================================== #
# do zrobienia:
# wiecej metryk do ewaluacji (np wariancja RAM i inne)
# automatyzacja ewaluacji
# zapisywanie wynikow do pliku
# tworzenie wykresow
# dodać ustawianie schedulera
# ======================================================== #


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

def evaluate(deploy_path, metric, scheduler):
    metrics = {
        "cpu_var": cpu_variance,
        "test": np.random.uniform
    }
    schedulers = {
        "ml": "custom-scheduler",
        "default": "default-scheduler"
    }
    if metric not in metrics:
        raise ValueError(f"Unknown metric: {metric}")
    if scheduler not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    values = []
    values.append(metrics[metric]())
    print("Before deployment:")
    print(f"CPU variance: {np.round(values[-1], 4)}")

    config.load_kube_config()
    cli = client.ApiClient()
    deploy_file = open(deploy_path, "r")
    deployment = yaml.safe_load(deploy_file)
    deploy_file.close()

    deployment["spec"]["template"]["spec"]["schedulerName"] = schedulers[scheduler]
    utils.create_from_dict(cli, deployment)

    values.append(metrics[metric]())
    print("After deployment:")
    print(f"CPU variance: {np.round(values[-1], 4)}")

    for i in range(EVALS):
        sleep(INTERVAL)
        values.append(metrics[metric]())
        print(f"After {(i + 1) * INTERVAL} seconds:")
        print(f"CPU variance: {np.round(values[-1], 4)}")

    log_file = open("logs/eval.json", "a", encoding="utf-8")
    data = {"timestamp": get_timestamp(), "scheduler": schedulers[scheduler]}

    for i in range(EVALS + 2):
        data[f"val{i}"] = values[i]

    log_file.write(json.dumps(data, ensure_ascii=False) + "\n")
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-deployment", required=True)
    parser.add_argument("-metric", default="test")
    parser.add_argument("-scheduler", default="default")
    args = parser.parse_args()
    
    evaluate(args.deployment, args.metric, args.scheduler)
    df = pd.read_json("logs/eval.json", lines=True)
    print(df)
