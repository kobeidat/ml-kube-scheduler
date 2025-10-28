import requests
from time import strftime, localtime
from config import PROM_URL, TIMESTAMP_FORMAT


def get_timestamp():
    return strftime(TIMESTAMP_FORMAT, localtime())

def log(msg):
    print(f"Custom-Scheduler {get_timestamp()}: {msg}")

def query_prometheus(query, url):
    try:
        r = requests.get(f"{url}/api/v1/query", params={'query': query})
        return r.json().get("data", {}).get("result", [])
    except Exception as e:
        log(f"Error querying Prometheus: {e}")
        return []

def query_prometheus_cpu(range_vector = 30, url = PROM_URL):
    if range_vector < 3:
        raise ValueError(f"Too narrow time range: {range_vector}m")
    
    query = "1 - (avg by (node) (irate(node_cpu_seconds_total{mode=\"idle\"}["
    query += f"{range_vector}m])))"
    return query_prometheus(query, url)

def query_prometheus_mem(url = PROM_URL):
    query = "1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)"
    return query_prometheus(query, url)
