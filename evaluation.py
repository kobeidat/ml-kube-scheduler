from kubernetes import config, client, utils
from prometheus import query_prometheus_cpu, get_timestamp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import json
import yaml
import argparse
from pathlib import Path


# ======================================================== #
# do zrobienia:
# wiecej metryk do ewaluacji (np wariancja RAM i inne)
# opcja porownania od razu kilku schedulerow
# usprawnienie formatu zapisu danych do pliku
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

def chart(values, metric, scheduler, deploy_path, timestamp: str):
    metric_titles = {
        "cpu_var": "CPU variance",
        "test": "TEST metric"
    }

    plt.plot(values, label=scheduler)
    plt.title(f"Kubernetes scheduler evaluation by {metric_titles[metric]}")
    plt.xlabel("Evaluations")
    plt.ylabel(metric_titles[metric])
    plt.legend()

    pod_name = deploy_path.split("/")[-1].split("\\")[-1].split(".")[0]
    timestamp = timestamp[5:]
    timestamp = timestamp.replace(":", "-")
    timestamp = timestamp.replace(" ", "_")
    plt.savefig(f"logs/{pod_name}-{timestamp}.png")

def evaluate(deploy_path, metric, scheduler):
    metrics_dict = {
        "cpu_var": cpu_variance,
        "test": np.random.uniform
    }
    schedulers_dict = {
        "ml": "custom-scheduler",
        "default": "default-scheduler"
    }

    if metric not in metrics_dict:
        raise ValueError(f"Unknown metric: {metric}")
    if scheduler not in schedulers_dict:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    values = []
    values.append(metrics_dict[metric]())
    print("Before deployment:")
    print(f"CPU variance: {np.round(values[-1], 4)}")

    # config.load_kube_config()
    # cli = client.ApiClient()
    deploy_file = open(deploy_path, "r")
    deployment = yaml.safe_load(deploy_file)
    deploy_file.close()

    deployment["spec"]["template"]["spec"]["schedulerName"] = schedulers_dict[scheduler]
    # utils.create_from_dict(cli, deployment)

    values.append(metrics_dict[metric]())
    print("After deployment:")
    print(f"CPU variance: {np.round(values[-1], 4)}")

    for i in range(EVALS):
        sleep(INTERVAL)
        values.append(metrics_dict[metric]())
        print(f"After {(i + 1) * INTERVAL} seconds:")
        print(f"CPU variance: {np.round(values[-1], 4)}")

    log_file = open("logs/eval.json", "a", encoding="utf-8")
    timestamp = get_timestamp()
    data = {"timestamp": timestamp, "scheduler": schedulers_dict[scheduler]}

    for i in range(EVALS + 2):
        data[f"val{i}"] = values[i]

    log_file.write(json.dumps(data, ensure_ascii=False) + "\n")
    log_file.close()
    chart(values, metric, schedulers_dict[scheduler], deploy_path, timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-deployment", required=True)
    parser.add_argument("-metric", default="test")
    parser.add_argument("-scheduler", default="default")
    args = parser.parse_args()
    
    evaluate(args.deployment, args.metric, args.scheduler)
    df = pd.read_json("logs/eval.json", lines=True)
    print(df)
