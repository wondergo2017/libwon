import torch.multiprocessing as mp
from queue import Queue
import time
import random
import os
import numpy as np

def dummy_func(dev, cfg):
    time.sleep(random.random() * 2)


def dummy_config():
    return list(range(20))

def mp_exec(resources, configs, func):
    """
    @ resources : list of gpu devices
    @ configs : list of params
    @ func : f(dev,cfg)
    """
    q = Queue()
    for res in resources:
        q.put(res)
    pool = mp.Pool()

    def put_back_dev(dev, cfg):
        def callback(*args):
            print(f"Device {dev} Finish cfg {cfg} ")
            q.put(dev)
            print(*args)

        return callback

    for idx, cfg in enumerate(configs):
        dev = q.get()
        print(f"Start config {cfg} on device {dev}")
        pool.apply_async(
            func,
            args=[dev, cfg],
            callback=put_back_dev(dev, cfg),
            error_callback=put_back_dev(dev, cfg),
        )

    pool.close()
    pool.join()


def mp_exec_trial(resources, configs, func, check_done, remove_run_flag, trial_time):
    """
    @ resources : list of gpu devices
    @ configs : list of params
    @ func : f(dev,cfg)
    @ check_done : f(cfg)
    @ remove_run_flag : f(cfg)
    """
    q = Queue()
    for res in resources:
        q.put(res)

    qcfg = Queue()
    for cfg in configs:
        qcfg.put(cfg)

    pool = mp.Pool()

    def put_back_dev(dev, cfg):
        def callback(*args):
            print(f"Device {dev} Finish cfg {cfg} ")
            if not check_done(cfg):
                remove_run_flag(cfg)
                print(f"Undone cfg {cfg} ")
                qcfg.put(cfg)
                sleeptime = int((random.random()) * trial_time)
                time.sleep(sleeptime)  # dev collsion, so wait to put current dev
            q.put(dev)
            print(*args)

        return callback

    while not qcfg.empty():
        cfg = qcfg.get()
        dev = q.get()
        print(f"Start config {cfg} on device {dev}")
        pool.apply_async(
            func,
            args=[dev, cfg],
            callback=put_back_dev(dev, cfg),
            error_callback=put_back_dev(dev, cfg),
        )

    pool.close()
    pool.join()


