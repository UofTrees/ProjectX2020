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
region_list = ['crin']
lr_list = [1e-4]
batch_size_list = [128]
seq_len_list = [100]
num_epochs_list = [100]
n_hidden_list = [64]
model_list = ['lstm', 'rnn']

'python train_baseline.py --region=cr --lr=0.001 --batch_size=256 --seq_len=100 --num_epochs=1 --n_hidden=20 --model_name=lstm'

# For each hyperparam combination, create an sbatch file to run
with open("train_all.sh", "w") as allf:

    for region in region_list:
        for lr in lr_list:
            for batch_size in batch_size_list:
                for seq_len in seq_len_list:
                    for num_epochs in num_epochs_list:
                        for n_hidden in n_hidden_list:
                            for model in model_list:

                                job = (
                                    f"{region}"
                                    + f"_model{model}"
                                    + f"_lr{lr:.1e}"
                                    + f"_batch{batch_size}"
                                    + f"_seq{seq_len}"
                                    + f"_epochs{num_epochs}"
                                    + f"_hidden{n_hidden}"
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
                                        + f"--region={region} "
                                        + f"--lr={lr} "
                                        + f"--batch_size={batch_size} "
                                        + f"--seq_len={seq_len} "
                                        + f"--num_epochs={num_epochs} "
                                        + f"--n_hidden={n_hidden} "
                                        + f"--model_name={model}")

                                allf.write(f"sbatch '{job_file}'\n")
