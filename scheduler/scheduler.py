import random
import json
from time import localtime, strftime
from kubernetes import client, config, watch
import requests

timestamp_format = "%Y-%m-%d %H:%M:%S"
scheduler_name = "custom-scheduler"
PROMETHEUS_URL = "http://prometheus-server.default.svc.cluster.local:9090"

def get_timestamp():
  return strftime(timestamp_format, localtime())

def query_prometheus(query):
  try:
    r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
    return r.json().get("data", {}).get("result", [])
  except Exception as e:
    print(f"Error querying Prometheus: {e}")
    return []

config.load_incluster_config()
v1 = client.CoreV1Api()

def nodes_available():
  ready_nodes = []
  for n in v1.list_node().items:
    for status in n.status.conditions:
      if status.status == "True" and status.type == "Ready":
        ready_nodes.append(n.metadata.name)
  return ready_nodes

def scheduler(pod_name, node, namespace="default"):
  target = client.V1ObjectReference()
  target.kind = "Node"
  target.apiVersion = "v1"
  target.name = node
  meta = client.V1ObjectMeta()
  meta.name = pod_name
  body = client.V1Binding(target=target)
  body.metadata = meta
  return v1.create_namespaced_binding(namespace, body, _preload_content=False)

def main():
  print("Custom-Scheduler: {}: Starting custom scheduler...".format(get_timestamp()))
  w = watch.Watch()
  for event in w.stream(v1.list_namespaced_pod, "default"):
    if event['object'].status.phase == "Pending" and event['object'].spec.scheduler_name == scheduler_name:
      try:
        pod_name = event['object'].metadata.name
        nodes_list = list(set(nodes_available()))
        random_node = random.choice(nodes_list)
        res = scheduler(pod_name, random_node)
        print("Custom-Scheduler: {}: Scheduling result: {}".format(get_timestamp(), res.status))
        print("Custom-Scheduler: {}: Scheduled {} to {}".format(get_timestamp(), pod_name, random_node))
      except client.rest.ApiException as e:
        print("Custom-Scheduler: {}: An exception occurred: {}".format(get_timestamp(), json.loads(e.body)['message']))

if __name__ == '__main__':
  main()
