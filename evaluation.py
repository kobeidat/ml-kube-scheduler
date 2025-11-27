from kubernetes import client, config
from kubernetes.client.rest import ApiException
from prometheus import query_prometheus_cpu, query_prometheus_mem
from prometheus import get_timestamp
from config import EVALS, INTERVAL, PROM_URL_EVAL, REVERSE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import json
import yaml
import argparse


def cpu_variance():
    response = query_prometheus_cpu(3, PROM_URL_EVAL)
    if not response:
        raise ConnectionError("Connection with Prometheus error")

    cpu_usage = []

    for node_data in response:
        cpu_usage.append(float(node_data["value"][1]))

    cpu_usage = np.array(cpu_usage) * 100
    var = np.var(cpu_usage)
    return var

def mem_variance():
    response = query_prometheus_mem(PROM_URL_EVAL)
    if not response:
        raise ConnectionError("Connection with Prometheus error")
    
    mem_usage = []

    for node_data in response:
        mem_usage.append(float(node_data["value"][1]))
    
    mem_usage = np.array(mem_usage) * 100
    var = np.var(mem_usage)
    return var

def save_graph(metric, deploy_path, timestamp):
    metric_titles = {
        "cpu_var": "CPU variance",
        "mem_var": "Memory variance",
        "test": "TEST metric"
    }

    plt.title(f"Kubernetes scheduler evaluation by {metric_titles[metric]}")
    plt.xlabel("Evaluations")
    plt.ylabel(metric_titles[metric])
    plt.legend()

    pod_name = deploy_path.split("/")[-1].split("\\")[-1].split(".")[0]
    timestamp = timestamp[5:]
    timestamp = timestamp.replace(":", "-")
    timestamp = timestamp.replace(" ", "_")
    plt.savefig(f"logs/{pod_name}-{timestamp}.png")

def evaluate(pod_paths, metric, scheduler, graph):
    metrics_dict = {
        "cpu_var": cpu_variance,
        "mem_var": mem_variance,
        "test": np.random.uniform
    }
    schedulers_dict = {
        "ml": "my-scheduler",
        "default": "default-scheduler"
    }

    if metric not in metrics_dict:
        raise ValueError(f"Unknown metric: {metric}")
    if scheduler not in schedulers_dict:
        raise ValueError(f"Unknown scheduler: {scheduler}")
    
    pods = []
    for path in pod_paths:
        pod_file = open(path, "r")
        pods.append(yaml.safe_load(pod_file))
        pod_file.close()

    config.load_kube_config()
    v1 = client.CoreV1Api()
    
    schedulers_list = [scheduler, "default"]
    if REVERSE:
        schedulers_list.reverse()

    for curr_scheduler in schedulers_list:
        print(f"\nSCHEDULER: {schedulers_dict[curr_scheduler]}")
        values = []

        values.append(metrics_dict[metric]())
        print("Before pods:")
        print(f"CPU variance: {np.round(values[-1], 4)}\n")

        for pod in pods:
            pod["spec"]["schedulerName"] = schedulers_dict[curr_scheduler]
            pod_namespace = pod["metadata"].get("namespace", "default")
            v1.create_namespaced_pod(pod_namespace, pod)
            print("Creating pod...", end="", flush=True)
            sleep(2)
            print("\nPod created")

        values.append(metrics_dict[metric]())
        print("\nAfter pods:")
        print(f"CPU variance: {np.round(values[-1], 4)}\n")

        for i in range(EVALS):
            sleep(INTERVAL)
            values.append(metrics_dict[metric]())
            print(f"After {(i + 1) * INTERVAL} seconds:")
            print(f"CPU variance: {np.round(values[-1], 4)}\n")
        
        for pod in pods:
            pod_name = pod["metadata"]["name"]
            pod_namespace = pod["metadata"].get("namespace", "default")
            v1.delete_namespaced_pod(pod_name, pod_namespace)
            print("Deleting pods...", end="", flush=True)

            while True:
                try:
                    v1.read_namespaced_pod(pod_name, pod_namespace)
                    sleep(1)
                except ApiException as e:
                    if e.status == 404:
                        break
                    else:
                        raise

            print("\nPods deleted")

        log_file = open("logs/eval.json", "a", encoding="utf-8")
        timestamp = get_timestamp()
        data = {
            "timestamp": timestamp,
            "scheduler": schedulers_dict[curr_scheduler]
        }

        for i in range(EVALS + 2):
            data[f"val{i}"] = values[i]

        log_file.write(json.dumps(data, ensure_ascii=False) + "\n")
        log_file.close()
        plt.plot(values, label=schedulers_dict[curr_scheduler])

    if graph:
        save_graph(metric, pod_paths[-1], timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pods", nargs="+", required=True)
    parser.add_argument("-scheduler", default="ml")
    parser.add_argument("-metric", default="test")
    parser.add_argument("-graph", action="store_true")
    args = parser.parse_args()

    if args.scheduler == "default":
        parser.error("Provide scheduler to compare it with the default one")

    evaluate(args.pods, args.metric, args.scheduler, args.graph)
    df = pd.read_json("logs/eval.json", lines=True)
    print(df)
