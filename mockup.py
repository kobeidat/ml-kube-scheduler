#!/usr/bin/env python3

import threading
import time
import random
import string
import torch
import torch.nn as nn
import numpy as np

NUM_FEATURES = 10
SEQ_LENGTH = 5
PREDICT_HORIZON = 3
UPDATE_INTERVAL = 5

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

feature_map = {}
reverse_feature_map = {}
input_history = []
model = LSTMModel(NUM_FEATURES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
lock = threading.Lock()

def generate_random_data():
    keys = random.sample(string.ascii_lowercase, random.randint(3, 10))
    return {k: random.random() for k in keys}

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
        data = generate_random_data()
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
        time.sleep(UPDATE_INTERVAL)

def predict_and_report():
    while True:
        time.sleep(UPDATE_INTERVAL / 2)
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
                letter = reverse_feature_map.get(min_idx, '?')
                print(f"Lowest projected average feature: {letter} (index {min_idx}) -> {avg_preds[min_idx]:.3f}")

update_thread = threading.Thread(target=model_updater, daemon=True)
update_thread.start()

predict_and_report()
