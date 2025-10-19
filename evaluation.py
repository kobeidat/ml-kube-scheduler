from kubernetes import config, client, utils
from prometheus import query_prometheus_cpu, get_timestamp
import numpy as np
from time import sleep
import json
import pandas as pd

# ================================================== #
# do zrobienia:
# metryki do ewaluacji (np wariancja RAM i inne)
# automatyzacja ewaluacji
# zapisywanie wynikow do pliku
# ================================================== #


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

    var1 = cpu_variance()
    print("Before deployment:")
    print("CPU variance: " + np.round(var1, 4))

    utils.create_from_yaml(client, "pods/cpu-stress-deploy.yaml")
    var2 = cpu_variance()
    print("After deployment:")
    print("CPU variance: " + np.round(var2, 4))

    sleep(60)
    var3 = cpu_variance()
    print("After 60 seconds:")
    print("CPU variance: " + np.round(var3, 4))

    file = open("logs/eval.json", "a", encoding="utf-8")
    data = {"timestamp": get_timestamp(), "var1": var1, "var2": var2, "var3": var3}
    file.write(json.dumps(data, ensure_ascii=False) + "\n")
    file.close()


if __name__ == "__main__":
    df = pd.read_json("logs/eval.json", lines=True)
    print(df)
