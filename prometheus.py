import requests
from time import strftime, localtime


PROMETHEUS_URL = "http://prometheus-server.default.svc.cluster.local"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_timestamp():
    return strftime(TIMESTAMP_FORMAT, localtime())

def log(msg):
    print(f"Custom-Scheduler {get_timestamp()}: {msg}")

def query_prometheus(query):
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
        return r.json().get("data", {}).get("result", [])
    except Exception as e:
        log(f"Error querying Prometheus: {e}")
        return []

def query_prometheus_cpu():
    return query_prometheus(
        """1 - (avg by (node) (irate(node_cpu_seconds_total{mode="idle"}[30m])))"""
    )

def query_prometheus_mem():
    return query_prometheus(
        "1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)"
    )
