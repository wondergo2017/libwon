from libwon import COLLECTOR
from libwon.utils import get_arg_dict, setup_seed
import os
from argparse import ArgumentParser
import json
parser = ArgumentParser()
parser.add_argument('--device_id', type = str, default = "0")
parser.add_argument('--log_dir', type = str, default = "../logs/tmp")
parser.add_argument('--info', type = str, default = "info")
parser.add_argument('--seed', type = int, default = 0)

args = parser.parse_args()

# pre logs
log_dir = args.log_dir
info_dict = get_arg_dict(args)
json.dump(info_dict, open(os.path.join(log_dir, 'args.json'),'w'),indent=2)
print(args)

# run
setup_seed(args.seed)
def run(args):
    COLLECTOR.add("info", args.info)
run(args)

# post logs
# COLLECTOR.add_GPU_MEM(args.device_id)
COLLECTOR.save_all_time()
COLLECTOR.save(os.path.join(log_dir,'collector.json'))