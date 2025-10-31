"Project configuration file."

# scheduler
LR = 0.01
MAX_NODES = 5
NUM_FEATURES = MAX_NODES * 2
SEQ_LENGTH = 5
PREDICT_HORIZON = 3
UPDATE_INTERVAL = 5
SCHEDULER_NAME = "my-scheduler"
VALUE_MAX = 1
DEFAULT_ALPHA = 0.7

# prometheus
PROM_URL = "http://prometheus-server.default.svc.cluster.local"
PROM_URL_EVAL = "http://localhost:9090"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# evaluation
EVALS = 4
INTERVAL = 3 # 10
