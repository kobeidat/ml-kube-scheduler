#!/usr/bin/env python

import json
from kubernetes import client, config, watch
import numpy as np
import random
import requests
import random
import string
import threading
from time import localtime, sleep, strftime
import torch
import torch.nn as nn

NUM_FEATURES = 10
SEQ_LENGTH = 5
PREDICT_HORIZON = 3
UPDATE_INTERVAL = 60
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
SCHEDULER_NAME = "my-scheduler"
PROMETHEUS_URL = "http://prometheus-server.default.svc.cluster.local"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

config.load_incluster_config()
v1 = client.CoreV1Api()

def get_timestamp():
  return strftime(TIMESTAMP_FORMAT, localtime())

def query_prometheus(query):
  try:
    r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
    return r.json().get("data", {}).get("result", [])
  except Exception as e:
    print(f"Custom-Scheduler: Error querying Prometheus: {e}")
    return []

def query_prometheus_cpu():
    return query_prometheus('100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[30m])) * 100)')

def nodes_available():
    ready_nodes = []
    for n in v1.list_node().items:
        for status in n.status.conditions:
            if status.status == "True" and status.type == "Ready":
                ready_nodes.append(n.metadata.name)
    return ready_nodes

feature_map = {}
reverse_feature_map = {}
input_history = []
model = LSTMModel(NUM_FEATURES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
lock = threading.Lock()

def get_data():
    nodes_list = list(set(nodes_available()))
    cpu_data = query_prometheus_cpu()
    return {
        nodes_list[i]: float(cpu_usage[i]['value'][1])
        for i in range(len(nodes_list))
    }

def assign_features(data):
    for k in data.keys():
        if k not in feature_map:
            available_features = set(range(NUM_FEATURES)) - set(reverse_feature_map.keys())
            if available_features:
                f_idx = available_features.pop()
                feature_map[k] = f_idx
                reverse_feature_map[f_idx] = k

    input_vector = [0.0] * NUM_FEATURES
    for k, v in data.items():
        if k in feature_map:
            input_vector[feature_map[k]] = v
    return input_vector

def model_updater():
    global input_history
    while True:
        data = get_data()
        with lock:
            input_vector = assign_features(data)
            input_history.append(input_vector)
            if len(input_history) > SEQ_LENGTH:
                input_history = input_history[-SEQ_LENGTH:]
                X = torch.tensor([input_history[:-1]], dtype=torch.float32).to(device)
                y = torch.tensor([input_history[-1]], dtype=torch.float32).to(device)
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
        sleep(UPDATE_INTERVAL)

def predict_node():
    with lock:
        if len(input_history) >= SEQ_LENGTH:
            X = torch.tensor([input_history[-SEQ_LENGTH:]], dtype=torch.float32).to(device)
            preds = []
            for _ in range(PREDICT_HORIZON):
                output = model(X)
                preds.append(output.detach().cpu().numpy())
                X = torch.cat([X[:, 1:, :], output.unsqueeze(1)], dim=1)
            preds = np.array(preds).squeeze()
            avg_preds = preds.mean(axis=0)
            min_idx = np.argmin(avg_preds)
            node = reverse_feature_map.get(min_idx)
            print(f"Custom-Scheduler: {get_timestamp()}: Lowest projected average feature: {node} (index {min_idx}) -> {avg_preds[min_idx]:.3f}")
            return node

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
        if event['object'].status.phase == "Pending" and event['object'].spec.scheduler_name == SCHEDULER_NAME:
            try:
                print(event['object'].metadata.__dict__)
                pod_name = event['object'].metadata.name
                print("...")
                data = get_data()
                print("...")
                print(f"Custom-Scheduler: data: {data}")
                print("...")
                node = predict_node() or random.choice(nodes_available())
                res = scheduler(pod_name, node)
                print("Custom-Scheduler: {}: Scheduling result: {}".format(get_timestamp(), res.status))
                print("Custom-Scheduler: {}: Scheduled {} to {}".format(get_timestamp(), pod_name, node))
            except client.rest.ApiException as e:
                print("Custom-Scheduler: {}: An exception occurred: {}".format(get_timestamp(), json.loads(e.body)['message']))


if __name__ == '__main__':
    update_thread = threading.Thread(target=model_updater, daemon=True)
    update_thread.start()
    main()
