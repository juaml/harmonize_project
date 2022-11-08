import os
from pathlib import Path

cwd = os.getcwd()
log_dir = Path(cwd) / "logs"
log_dir.mkdir(exist_ok=True)

env = "juharmonize"

data_dir = "/data/project/harmonize/data/CAT/s4_r8/"
save_dir = Path("/data/project/harmonize/results_kbest/")


exp_params = {
    "sites_use": "eNKI ID1000 CamCAN 1000Gehirne",
    "pred_model": "rvr",
    "problem_type": "regression",
    "n_high_var_feats": -1,
    "scaler": None,
    "use_disk": None,
}

select_ks = list(range(100, 3401, 100))

experiments = {}
for k in select_ks:
    experiments[f"overfit_all_regression_{k}"] = exp_params.copy()
    experiments[f"overfit_all_regression_{k}"]["select_k"] = k

harmonize_modes = [
    ["cheat", "16G", 1, 0],
    # ["none", "16G", 1, 0],
    # ["target", "16G", 1, 0],
    # ["notarget", "16G", 1, 0],
    ["pretend", "16G", 1, 0],
    # ["pretend_nosite", "16G", 1, 0],
    # ["predict", "20G", 10, "1500G"],
    # ["predict_pretend", "20G", 10, "1500G"],
    # ["predict_pretend_nosite", "20G", 10, "1500G"],
]
n_splits = 5
n_repeats = 10


exec_name = (
    "run_all_data_parallel_cv.py "
    f"--data_dir {data_dir} "
    f"--save_dir {save_dir.as_posix()}/$(exp_name) "
    f"--n_splits {n_splits} "
    f"--n_repeats {n_repeats} "
    "--fold $(fold) "
    "--harmonize_mode $(harmonize_mode) "
    "--n_jobs $(cpus) "
)

log_suffix = "juharmonize_$(exp_name)/$(harmonize_mode)_$(fold)"

preamble = f"""
# The environment
universe       = vanilla
getenv         = True

# Resources
request_cpus   = $(cpus)
request_memory = $(memory)
request_disk   = $(disk)

# Executable
initial_dir    = {cwd}
executable     = {cwd}/run_in_venv.sh
transfer_executable = False

arguments      = {env} python {exec_name} $(args)

# Logs
log            = {log_dir.as_posix()}/{log_suffix}.log
output         = {log_dir.as_posix()}/{log_suffix}.out
error          = {log_dir.as_posix()}/{log_suffix}.err

"""


with open("all_data_kbest_parallel_cv.submit", "w") as f:
    f.writelines(preamble)
    for exp_name, exp_config in experiments.items():
        args = " ".join(
            f"--{arg_name} {arg_val}"
            if arg_val is not None
            else f"--{arg_name}"
            for arg_name, arg_val in exp_config.items()
        )

        t_log_dir = log_dir / f"juharmonize_{exp_name}"
        t_log_dir.mkdir(exist_ok=True, parents=True)
        t_save_dir = save_dir / exp_name
        t_save_dir.mkdir(exist_ok=True, parents=True)
        for t_mode, memory, cpus, disk in harmonize_modes:
            for i_fold in range(n_splits * n_repeats):
                f.write(f"exp_name={exp_name}\n")
                f.write(f"memory={memory}\n")
                f.write(f"cpus={cpus}\n")
                f.write(f"disk={disk}\n")
                f.write(f"args={args}\n")
                f.write(f"harmonize_mode={t_mode}\n")
                f.write(f"fold={i_fold}\n")
                f.write("queue\n\n")
