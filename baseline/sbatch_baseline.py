#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import pathlib

# Constants & set up
gres = "gpu:1"
QOS = "normal"
CPU = 4
RAM = "16GB"
partition = "rtx6000"

root = pathlib.Path(".").resolve()
job_dir = root / "jobs"
if not job_dir.exists():
    job_dir.mkdir()


# Hyperparams to try
lr_list = [1e-4]
hidden_dims_list = [20]
seq_len_list = [100, 70, 50, 30]
model_list = ['lstm', 'rnn']

# For each hyperparam combination, create an sbatch file to run
with open("train_all.sh", "w") as allf:

    for lr in lr_list:
        for hidden_dims in hidden_dims_list:
            for seq_len in seq_len_list:
                for model in model_list:

                    job = (
                        f"lr{lr:.1e}"
                        + f"_model{model}"
                        + f"_hidden{hidden_dims}"
                        + f"_seq_len{seq_len}"
                    )

                    job_file = job_dir / f"{job}.job"
                    job_out_file = job_dir / f"{job}.out"

                    with open(job_file, "w") as f:
                        f.write(f"#!/bin/bash\n")
                        f.write(f"#SBATCH -N 1\n")
                        f.write(f"#SBATCH -n 1\n")
                        f.write(f"#SBATCH --gres={gres}\n")
                        f.write(f"#SBATCH --qos={QOS}\n")
                        f.write(f"#SBATCH -p {partition}\n")
                        f.write(f"#SBATCH --cpus-per-task={CPU}\n")
                        f.write(f"#SBATCH --mem={RAM}\n")
                        f.write(f"#SBATCH --job-name='{job}'\n")
                        f.write(
                            f"#SBATCH --output='{job_out_file}'\n"
                        )
                        f.write(f"cd {root}\n")
                        f.write(
                            f"python3 train_baseline.py "
                            + f"--lr={lr} "
                            + f"--n_hidden={hidden_dims} "
                            + f"--seq_len={seq_len} "
                            + f"--model_name={model}")

                    allf.write(f"sbatch '{job_file}'\n")
