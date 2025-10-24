from kubernetes import config, client, utils
from prometheus import query_prometheus_cpu, query_prometheus_mem, get_timestamp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import json
import yaml
import argparse


# =================================================================== #
# do zrobienia:
# wiecej metryk np. latency pending time, sezonowosc obciazen, koszty
# ------------------------------------------------------------------- #
# wiecej roznych yaml odpalanych w jednym evaluation
# przetestowac napisane deploymenty
# opcja wylaczenia wykresow
# opcja puszczenia kilku roznych testow naraz
# wiecej scenariuszy testowych
# =================================================================== #


EVALS = 4
INTERVAL = 1

def cpu_variance():
    response = query_prometheus_cpu(1)
    cpu_usage = []

    for node_data in response:
        cpu_usage.append(float(node_data["value"][1]))

    cpu_usage = np.array(cpu_usage) * 100
    var = np.var(cpu_usage)
    return var

def mem_variance():
    response = query_prometheus_mem()
    mem_usage = []

    for node_data in response:
        mem_usage.append(float(node_data["value"][1]))
    
    mem_usage = np.array(mem_usage) * 100
    var = np.var(mem_usage)
    return var

def save_chart(metric, deploy_path, timestamp):
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

def evaluate(deploy_path, metric, scheduler):
    metrics_dict = {
        "cpu_var": cpu_variance,
        "mem_var": mem_variance,
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
    
    deploy_file = open(deploy_path, "r")
    deployment = yaml.safe_load(deploy_file)
    deploy_file.close()

    config.load_kube_config()
    cli = client.ApiClient()
    
    for curr_scheduler in [scheduler, "default"]:
        print(f"SCHEDULER: {schedulers_dict[curr_scheduler]}")
        values = []

        values.append(metrics_dict[metric]())
        print("Before deployment:")
        print(f"CPU variance: {np.round(values[-1], 4)}\n")

        deployment_spec = deployment["spec"]["template"]["spec"]
        deployment_spec["schedulerName"] = schedulers_dict[curr_scheduler]
        utils.create_from_dict(cli, deployment)
        print("Creating deployment...", end="", flush=True)
        sleep(2)
        print("\nDeployment created\n")

        values.append(metrics_dict[metric]())
        print("After deployment:")
        print(f"CPU variance: {np.round(values[-1], 4)}\n")

        for i in range(EVALS):
            sleep(INTERVAL)
            values.append(metrics_dict[metric]())
            print(f"After {(i + 1) * INTERVAL} seconds:")
            print(f"CPU variance: {np.round(values[-1], 4)}\n")
        
        pod_name = deployment["metadata"]["name"]
        pod_namespace = deployment["metadata"].get("namespace", "default")
        apps_v1 = client.AppsV1Api(cli)
        apps_v1.delete_namespaced_deployment(pod_name, pod_namespace)
        print("Deleting deployment...", end="", flush=True)
        sleep(10)
        print("\nDeployment deleted\n")

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

    save_chart(metric, deploy_path, timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-deployment", required=True)
    parser.add_argument("-scheduler", default="ml")
    parser.add_argument("-metric", default="test")
    args = parser.parse_args()

    if args.scheduler == "default":
        parser.error("Provide scheduler to compare it with the default one")

    evaluate(args.deployment, args.metric, args.scheduler)
    df = pd.read_json("logs/eval.json", lines=True)
    print(df)
