#!/usr/bin/env python

from kubernetes import client, watch, config as kube_config
import numpy as np
import random
import random
import threading
from time import sleep
import torch
import torch.nn as nn

from prometheus import query_prometheus_cpu, query_prometheus_mem, log
import config
from model import LSTMModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use load_incluster_config when deploying scheduler from within the cluster.
# Otherwise use load_kube_config
kube_config.load_incluster_config()
v1 = client.CoreV1Api()

def nodes_available():
    ready_nodes = []
    for n in v1.list_node().items:
        # This loops over the nodes available. n is the node. We are trying
        # to schedule the pod on one of those nodes.
        for status in n.status.conditions:
            if status.status == "True" and status.type == "Ready":
                ready_nodes.append(n.metadata.name)
    return ready_nodes

node_id_map = {}
reverse_node_id_map = {}
input_history = []
model = LSTMModel(config.NUM_FEATURES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
criterion = nn.MSELoss()
lock = threading.Lock()

def get_data():
    nodes_list = list(set(nodes_available()))
    cpu_data = query_prometheus_cpu()
    mem_data = query_prometheus_mem()
    return {
        nodes_list[i]: [
            float(cpu_data[i]['value'][1]),
            float(mem_data[i]['value'][1])
        ]
        for i in range(len(nodes_list))
    }

def assign_features(data):
    for k in data.keys():
        if k not in node_id_map:
            all_nodes = set(range(config.MAX_NODES))
            used_nodes = set(reverse_node_id_map.keys())
            available_nodes = all_nodes - used_nodes
            if available_nodes:
                idx = available_nodes.pop()
                node_id_map[k] = idx
                reverse_node_id_map[idx] = k

    input_vector = [config.VALUE_MAX for _ in range(config.NUM_FEATURES)]
    for k, vs in data.items():
        if k in node_id_map:
            input_vector[node_id_map[k]] = vs[0]
            input_vector[config.MAX_NODES + node_id_map[k]] = vs[1]
    return input_vector

def model_updater():
    global input_history
    while True:
        data = get_data()
        with lock:
            input_vector = assign_features(data)
            if len(input_history) > config.SEQ_LENGTH:
                input_history = input_history[-config.SEQ_LENGTH:]
                X = torch.tensor([input_history], dtype=torch.float32) \
                         .to(device)
                y = torch.tensor([input_vector], dtype=torch.float32) \
                         .to(device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
            input_history.append(input_vector)
        sleep(config.UPDATE_INTERVAL)

def predict_node(alpha):
    with lock:
        while len(input_history) < config.SEQ_LENGTH:
            log("Waiting for the inputs to gather...")
            sleep(config.UPDATE_INTERVAL)
        X = torch.tensor([input_history[-config.SEQ_LENGTH:]], dtype=torch.float32) \
                 .to(device)
        pred_vector = model(X).detach().cpu().numpy().squeeze()
        pred_cpu = pred_vector[:config.MAX_NODES]
        pred_mem = pred_vector[config.MAX_NODES:]
        scores = alpha * pred_cpu + (1 - alpha) * pred_mem
        min_idx = np.argmin(scores)
        log(f"pred {pred_vector}")
        log(f"scores {scores}")
        node = reverse_node_id_map.get(min_idx)
        log(f"Lowest projected average feature: {node} "
            + f"(index {min_idx}) -> "
            + f"{scores[min_idx]:.3f} ("
            + f"cpu: {pred_cpu[min_idx]:.3f}, "
            + f"mem: {pred_mem[min_idx]:.3f})")
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
    return v1.create_namespaced_binding(
        namespace,
        body,
        _preload_content=False
    )

def main():
    log("Starting custom scheduler...")
    w = watch.Watch()

    for event in w.stream(v1.list_namespaced_pod, "default"):
        pending = event['object'].status.phase == "Pending"
        this_scheduler = event['object'].spec.scheduler_name == config.SCHEDULER_NAME
        existing_node_name = event['object'].spec.node_name
        if pending and this_scheduler:
            if existing_node_name:
                log(f"Pod already bound to {existing_node_name}")
                continue
            try:
                annotations = event['object'].metadata.annotations or {}
                alpha_str = annotations.get(
                    f'{config.SCHEDULER_NAME}/alpha',
                    str(config.DEFAULT_ALPHA)
                )
                alpha = float(alpha_str) or config.DEFAULT_ALPHA
                log(f"Alpha: {alpha}")
                pod_name = event['object'].metadata.name
                log(f"Pod: {pod_name}")
                data = get_data()
                log(f"data: {data}")
                node = predict_node(alpha) or random.choice(nodes_available())
                res = scheduler(pod_name, node)
                log(f"Scheduling result: {res.status}")
                log(f"Scheduled {pod_name} to {node}")
            except client.rest.ApiException as e:
                log("An exception occurred: {json.loads(e.body)['message']}")


if __name__ == '__main__':
    update_thread = threading.Thread(target=model_updater, daemon=True)
    update_thread.start()
    main()
