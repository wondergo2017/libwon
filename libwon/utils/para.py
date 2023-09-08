import os
from .mp import mp_exec, mp_exec_trial
from .collector import COLLECTOR
from .cloud import send_autodl
from .misc import *
class ParallelerGrid:
    def __init__(
        self,
        gpus,
        grid_list,
        log_dir,
        cmd,
        readme="",
        log_arg="save",
        gpu_arg="gpu",
        exp_script_dir=None,
        base_script_dir=None,
        finish_file="collector.json",
        phone_notice=None,
        epoch_arg="p_epoch",
        trial=False,
        trial_time=30,
    ):
        self.resources = gpus
        self.grid_list = grid_list
        self.log_dir = log_dir
        self.cmd = cmd
        self.log_arg = log_arg
        self.gpu_arg = gpu_arg
        self.exp_script_dir = exp_script_dir
        self.base_script_dir = base_script_dir
        self.readme = readme
        self.finish_file = finish_file
        self.phone_notice = phone_notice
        self.epoch_arg = epoch_arg
        self.trial = trial
        self.trial_time = trial_time

        self.exp_dir = os.path.join(log_dir, "exp")
        self.ana_dir = os.path.join(log_dir, "ana")
        self.scr_dir = os.path.join(log_dir, "scripts")

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ana_dir, exist_ok=True)
        os.makedirs(self.scr_dir, exist_ok=True)

        open(os.path.join(self.log_dir, "readme.md"), "w").write(self.readme)

    @staticmethod
    def collect_keys(grid_list):
        if isinstance(grid_list, list):
            s = set()
            for l in grid_list:
                s = s.union(set(ParallelerGrid.collect_keys(l)))
            return sorted(list(s))
        elif isinstance(grid_list, dict):
            return list(grid_list.keys())
        else:
            assert False

    def get_configs(self):
        grid_list = self.grid_list

        def gen_config_one(grid_list):
            configs = []
            config = {}
            param_names = list(grid_list.keys())

            def gen_config(ni, config, configs):
                if ni >= len(param_names):
                    configs.append(config.copy())
                    return
                pname = param_names[ni]
                plist = grid_list[pname]
                for i, p in enumerate(plist):
                    config[pname] = p
                    gen_config(ni + 1, config, configs)
                    del config[pname]

            gen_config(0, config, configs)
            return configs

        configs = []
        if isinstance(grid_list, list):
            for g in grid_list:
                configs.extend(gen_config_one(g))
        else:
            configs.extend(gen_config_one(grid_list))
        return configs

    def cfg2dirname(self, cfg):
        folder = "_".join(list(map(str, cfg.values())))
        return folder

    def check_done(self, cfg):
        folder = self.cfg2dirname(cfg)
        log_dir = os.path.join(self.exp_dir, folder)
        done = os.path.join(log_dir, self.finish_file)
        return os.path.exists(done)

    def remove_run_flag(self, cfg):
        folder = self.cfg2dirname(cfg)
        log_dir = os.path.join(self.exp_dir, folder)
        cmd = os.path.join(log_dir, "cmd.txt")
        if os.path.exists(cmd):
            os.remove(cmd)

    def get_run_time(self, configs):
        times = []
        for cfg in configs:
            folder = self.cfg2dirname(cfg)
            try:
                COLLECTOR.load(os.path.join(self.exp_dir, folder, "collector.json"))
                time = COLLECTOR.cache["all_time"][0]
                times.append(float(time))
            except:
                pass
        return times

    def check_finish(self, detail=False):
        configs = self.get_configs()
        finish = []
        unfinish = []
        running = []
        for idx, cfg in enumerate(configs):
            folder = self.cfg2dirname(cfg)
            log_dir = os.path.join(self.exp_dir, folder)
            cmd = os.path.join(log_dir, "cmd.txt")
            done = os.path.join(log_dir, self.finish_file)
            if os.path.exists(done):
                finish.append([idx, cfg])
            else:
                if not os.path.exists(cmd):
                    unfinish.append([idx, cfg])
                else:
                    running.append([idx, cfg])
        if detail:
            print("#" * 30, "Finish", "#" * 30)
            for idx, cfg in finish:
                print(idx, cfg)

            print("#" * 30, "UnFinish", "#" * 30)
            for idx, cfg in unfinish:
                print(idx, cfg)

            print("#" * 30, "Running", "#" * 30)
            for idx, cfg in running:
                print(idx, cfg)
        times = self.get_run_time(configs)
        time_avg = convert_time(np.mean(times)) if len(times) else "NaN"
        time_till_now = (
            convert_time(sum(times) / len(self.resources)) if len(times) else "NaN"
        )
        time_to_go = (
            convert_time(
                (np.mean(times) * (len(unfinish) + len(running))) / len(self.resources)
            )
            if len(times)
            else "NaN"
        )
        print(
            f"# Total={len(configs)} # Finish={len(finish)} , # Unfinish={len(unfinish)} , # Running={len(running)}, # TotalTime={time_till_now}, # AvgTime={time_avg}, # TimeToGo={time_to_go}"
        )
        for dir in [self.ana_dir, self.exp_dir, self.scr_dir]:
            print(
                f"DIR {os.path.abspath(dir)} Using {convert_size(count_dir_size(dir))}"
            )
        return finish, unfinish

    def execute(self, cp=True):
        from argparse import ArgumentParser

        def get_args(args=None):
            parser = ArgumentParser()
            parser.add_argument(
                "-t",
                type=str,
                default="show",
                choices=["show", "run", "debug", "clear", "check"],
            )
            parser.add_argument("-c", type=int, default=0)
            parser.add_argument("-f", type=int, default=0)

            args = parser.parse_args(args)
            return args

        args = get_args()
        t = args.t
        c = args.c
        self.f = args.f

        if cp:
            if t in "run debug".split():
                cp_pys(self.exp_script_dir, self.scr_dir)
                cp_pys(self.base_script_dir, self.scr_dir)

        if t == "show":
            self.show(c)
        elif t == "run":
            self.run()
            if self.phone_notice:
                send_autodl(**self.phone_notice)
        elif t == "debug":
            self.debug()
        elif t == "clear":
            self.clear()
        elif t == "check":
            self.check_finish(True)

    def func(self, dev, cfg):
        folder = self.cfg2dirname(cfg)
        log_dir = os.path.join(self.exp_dir, folder)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "log_out.txt")

        cmd = f'{self.cmd} --{self.gpu_arg} {dev} --{self.log_arg} "{log_dir}" > "{log_file}" '
        for pname, value in cfg.items():
            cmd += f" --{pname} {value}"
        open(os.path.join(log_dir, "cmd.txt"), "w").write(cmd)
        print("CMD ", cmd)

        os.system(cmd)

    def show(self):
        pass

    def debug(self):
        cfg = self.get_configs()[0]
        cfg[self.epoch_arg] = 2
        self.func(self.resources[0], cfg)

    def run(self):
        configs = self.get_configs()
        finish, _ = self.check_finish(False)
        idxs = sorted(
            list(set(list(range(len(configs)))) - set([x[0] for x in finish]))
        )
        print("config idxs to run: ", idxs)
        if not self.f:
            configs = [configs[i] for i in idxs]
        if not self.trial:
            mp_exec(self.resources, configs, self.func)
        else:
            mp_exec_trial(
                self.resources,
                configs,
                self.func,
                self.check_done,
                self.remove_run_flag,
                self.trial_time,
            )

    def clear(self):
        log_dir = os.path.abspath(self.log_dir)
        print(f"Removing {log_dir} ? y or n ")
        a = input()
        if a.strip().lower() == "y":
            print(f"Removing {log_dir}")
            shutil.rmtree(log_dir)
