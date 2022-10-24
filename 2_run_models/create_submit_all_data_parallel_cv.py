import os
from pathlib import Path

cwd = os.getcwd()
log_dir = Path(cwd) / "logs"
log_dir.mkdir(exist_ok=True)

env = "juharmonize"

data_dir = "/data/project/harmonize/data/CAT/s4_r4/"
save_dir = Path("/data/project/harmonize/results/")

experiments = {
    "test_all_regression": {
        "sites_use": "eNKI ID1000 CamCAN 1000Gehirne",
        "pred_model": "rvr",
        "problem_type": "regression",
        "n_high_var_feats": 29,
        "scaler": None,
        "unify_sites" : ""
    },

}

harmonize_modes = [
    "cheat",
    "none",
    "target",
    "notarget",
    "predict",
    "pretend",
    "pretend_nosite",
    "predict_pretend",
    "predict_pretend_nosite",
]
n_splits = 5


exec_name = (
    "run_all_data_parallel_cv.py "
    f"--data_dir {data_dir} "
    f"--save_dir {save_dir.as_posix()}/$(exp_name) "
    "--n_splits $(n_splits) "
    "--fold $(fold) "
    "--harmonize_mode $(harmonize_mode) "
)

log_suffix = "juharmonize_$(exp_name)/$(harmonize_mode)_$(fold)"

preamble = f"""
# The environment
universe       = vanilla
getenv         = True

# Resources
request_cpus   = 1
request_memory = 16G
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


with open("all_data_parallel_cv.submit", "w") as f:
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
        for t_mode in harmonize_modes:
            for i_fold in range(n_splits):
                f.write(f"exp_name={exp_name}\n")
                f.write(f"args={args}\n")
                f.write(f"harmonize_mode={t_mode}\n")
                f.write(f"fold={i_fold}\n")
                f.write(f"n_splits={n_splits}\n")
                f.write("queue\n\n")
