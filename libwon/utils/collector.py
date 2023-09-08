import json
import torch
import numpy as np
import time
class Collector:
    def __init__(self):
        self.cache = {}
        self.init_time = time.time()
        self.mute = False

    def value2str(self,value):
        if isinstance(value, float):
            return round(value, 4)
        elif isinstance(value, list):
            return [self.value2str(x) for x in value]
        elif isinstance(value, dict):
            return {k: self.value2str(x) for k, x in value.items()}
        return f"{value}"
    
    def __getitem__(self, key):
        return self.cache[key]

    def add(self, key, value):
        cache = self.cache
        if key not in cache:
            cache[key] = []
        cache[key].append(value)
        epoch = len(cache[key]) - 1
        value = self.value2str(value)
        if not self.mute:
            print(f"COLLECTOR Epoch {epoch:03d} : {key}={value}")

    def save(self, path):
        json.dump(self.cache, open(path, "w"))

    def load(self, path):
        self.cache = json.load(open(path))

    def clear(self):
        self.cache = {}

    def add_GPU_MEM(self, device, id = True):
        if id:
            device = f"cuda:{device}"
        t = torch.cuda.get_device_properties(device).total_memory
        r = torch.cuda.memory_reserved(device)
        a = torch.cuda.memory_allocated(device)
        self.add("GPU_MEM_total_MB", t / (1024**2))
        self.add("GPU_MEM_reserved_MB", r / (1024**2))
        self.add("GPU_MEM_allocated_MB", a / (1024**2))

    def add_graph_data(self, dataset):
        self.add("#graphs", len(dataset))
        self.add("avg_nodes", np.mean([d.x.shape[0] for d in dataset]))
        self.add("avg_edges", np.mean([d.edge_index.shape[1] for d in dataset]))
        self.add("feat_dim", dataset[0].x.shape[1])

    def add_node_data(self, dataset):
        self.add("nodes", dataset.x.shape[0])
        self.add("edges", dataset.edge_index.shape[1])
        self.add("feat_dim", dataset.x.shape[1])
        self.add("num_classes", torch.max(dataset.y).item() + 1)

    def get_single(self):
        d = {}
        for k, v in self.cache.items():
            if len(v) == 0:
                d[k] = v[0]
        return d

    def get_seq(self):
        d = {}
        for k, v in self.cache.items():
            if len(v) > 1:
                d[k] = v
        return d

    def save_all_time(self):
        self.add("all_time", time.time() - self.init_time)


COLLECTOR = Collector()
