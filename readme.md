# libwon
A simple lib to automatically manage the parallelled experiment details of different configs. It records the internal results, params, etc.

## Usage

### Collector

It is a global object to collect results, configs, etc.
```python
from libwon import COLLECTOR
COLLECTOR.mute = False          # whether logging when adding values
COLLECTOR.add("info", "XXX")    # add key as info, a value of "XXX" appended to the list  
COLLECTOR.add_GPU_MEM("cuda:0") # save memory usage of cuda:0
COLLECTOR.save_all_time()       # save executing time till now
COLLECTOR.save(os.path.join(log_dir,'collector.json'))  # save the collector
```
### ParallelGrid
It is a class to 
- generate configs from grid lists
- execute programs in parallel with given gpu resources
- automatically save logs, scripts, configs, results, etc., for reproducibility
- automatically fetch logs, configs, etc., for analyses

``` python
import os
from libwon import COLLECTOR, ParallelerGrid
import pandas as pd
import json

grid_list = [   # the config grid list
    {"seed": range(3), "info":[1]},
    {"seed": [0], "info":range(3)},
]

gpus = [0, 0, 1, 2]     # the available gpu resources    

cmd = "python main.py"  # the program entry file 
readme = "test"
phone_notice = None     # phone notice if finishing the run.
gpu_arg, log_arg = "device_id", "log_dir"   # arg flags in cmd.
base_script_dir = "./"                      # the program script dir to be saved
dir, fname = os.path.abspath(__file__).split(os.sep)[-2:]
log_dir = os.path.join("../../logs/", dir, os.path.splitext(fname)[0])

class Parallel(ParallelerGrid):
    # overwrite the functions to customize
    def show(self, c):  
        exp_dir = self.exp_dir
        ana_dir = self.ana_dir
        cols = ParallelerGrid.collect_keys(grid_list)
        print("Columns", cols)
        files = os.listdir(exp_dir)
        res = {}
        for fname in files:
            folder = os.path.join(exp_dir, fname)
            info = json.load(open(os.path.join(folder, "args.json")))
            hp = tuple([info[k] for k in cols])
            try:
                COLLECTOR.load(os.path.join(exp_dir, fname, "collector.json"))
                cache = COLLECTOR.cache
                res[hp] = cache
            except Exception as e:
                print(e)

        if c == 0:
            table = []
            metrics = ['info']
            for k,v in res.items():
                table.append([*k]+[v[m][0] for m in metrics])
            df = pd.DataFrame(table, columns= cols + metrics)
            print(df)

parallel = Parallel(
    gpus = gpus,
    log_dir = log_dir,
    grid_list = grid_list,
    cmd = cmd,
    gpu_arg = gpu_arg,
    log_arg = log_arg,
    exp_script_dir = os.path.dirname(os.path.abspath(__file__)),
    base_script_dir = base_script_dir,
    readme = readme,
    phone_notice = phone_notice,
).execute()

```