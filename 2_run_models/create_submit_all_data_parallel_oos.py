import os
from pathlib import Path

cwd = os.getcwd()
log_dir = Path(cwd) / "logs"
log_dir.mkdir(exist_ok=True)

env = "juharmonize"

data_dir = "/data/project/harmonize/data/CAT/s4_r4/"
save_dir = Path("/data/project/harmonize/results/")

experiments = {
    "test_all_big_oos_eNKI": {
        "sites_use": "ID1000 CamCAN 1000Gehirne",
        "sites_oos": "eNKI",
        "pred_model": "rvr",
        "problem_type": "regression",
        "n_high_var_feats": 29854,
        "scaler": None,
        "unify_sites": "None",
    },
    "test_all_big_oos_ID1000": {
        "sites_use": "eNKI CamCAN 1000Gehirne",
        "sites_oos": "ID1000",
        "pred_model": "rvr",
        "problem_type": "regression",
        "n_high_var_feats": 29854,
        "scaler": None,
        "unify_sites": "None",
    },
    "test_all_big_oos_CamCAN": {
        "sites_use": "eNKI ID1000 1000Gehirne",
        "sites_oos": "CamCAN",
        "pred_model": "rvr",
        "problem_type": "regression",
        "n_high_var_feats": 29854,
        "scaler": None,
        "unify_sites": "None",
    },
    "test_all_big_oos_1000Gehirne": {
        "sites_use": "eNKI ID1000 CamCAN",
        "pred_model": "rvr",
        "sites_oos": "1000Gehirne",
        "problem_type": "regression",
        "n_high_var_feats": 29854,
        "scaler": None,
        "unify_sites": "None",
    },
}

harmonize_modes = [
    ["target", "16G"],
    ["notarget", "16G"],
    ["predict", "16G"],
    ["pretend", "16G"],
    ["pretend_nosite", "16G"],
    ["predict_pretend", "500G"],
    ["predict_pretend_nosite", "500G"],
]

exec_name = (
    "run_all_data_parallel_oos.py "
    f"--data_dir {data_dir} "
    f"--save_dir {save_dir.as_posix()}/$(exp_name) "
    "--harmonize_mode $(harmonize_mode) "
)

log_suffix = "juharmonize_$(exp_name)/$(harmonize_mode)"

preamble = f"""
# The environment
universe       = vanilla
getenv         = True

# Resources
request_cpus   = 1
request_memory = $(memory)
request_disk   = 0

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


with open("all_data_parallel_oos.submit", "w") as f:
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
        for t_mode, memory in harmonize_modes:
            f.write(f"exp_name={exp_name}\n")
            f.write(f"memory={memory}\n")
            f.write(f"args={args}\n")
            f.write(f"harmonize_mode={t_mode}\n")
            f.write("queue\n\n")
