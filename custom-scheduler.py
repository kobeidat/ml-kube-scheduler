#!/usr/bin/env python

import random
import json
from time import localtime, strftime
from kubernetes import client, config, watch
import requests

timestamp_format = "%Y-%m-%d %H:%M:%S"
scheduler_name = "my-scheduler"
prometheus_url = "http://prometheus-server.default.svc.cluster.local:9090"

# Use load_incluster_config when deploying scheduler from within the cluster. Otherwise use load_kube_config
config.load_incluster_config()
v1 = client.CoreV1Api()

def get_timestamp():
  return strftime(timestamp_format, localtime())

def query_prometheus(query):
  try:
    r = requests.get(f"{prometheus_url}/api/v1/query", params={'query': query})
    print(f"Resp: {r.json()}")
    return r.json().get("data", {}).get("result", [])
  except Exception as e:
    print(f"Custom-Scheduler: Error querying Prometheus: {e}")
    return []

def get_timestamp():
    return strftime(timestamp_format, localtime())

def nodes_available():
    ready_nodes = []
    for n in v1.list_node().items:
        # This loops over the nodes available. n is the node. We are trying to schedule the pod on one of those nodes.
        for status in n.status.conditions:
            if status.status == "True" and status.type == "Ready":
                ready_nodes.append(n.metadata.name)
    return ready_nodes


# You can use "default" as a namespace.
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
        # We get an "event" whenever a pod needs to be scheduled
        # and event['object'].spec.scheduler_name == scheduler_name:
        if event['object'].status.phase == "Pending" and event['object'].spec.scheduler_name == scheduler_name:
            try:
                print(event['object'].metadata.__dict__)
                pod_name = event['object'].metadata.name
                print("...")
                prometheus_data = query_prometheus('100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[30m])) * 100)')
                print("...")
                print(f"Custom-Scheduler: Prometheus data: {prometheus_data}")
                print("...")
                nodes_list = list(set(nodes_available()))
                print("...")
                print(f"Custom-Scheduler: Nodes: {nodes_list}")
                random_node = random.choice(nodes_list)
                res = scheduler(pod_name, random_node)
                print("Custom-Scheduler: {}: Scheduling result: {}".format(get_timestamp(), res.status))
                print("Custom-Scheduler: {}: Scheduled {} to {}".format(get_timestamp(), pod_name, random_node))
            except client.rest.ApiException as e:
                print("Custom-Scheduler: {}: An exception occurred: {}".format(get_timestamp(), json.loads(e.body)['message']))


if __name__ == '__main__':
    main()
