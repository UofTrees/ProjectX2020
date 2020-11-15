#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import pathlib

# Constants & set up
gres = "gpu:1"
QOS = "normal"
CPU = 4
RAM = "16GB"
partition = "gpu"

root = pathlib.Path(".").resolve()
job_dir = root / "jobs"
if not job_dir.exists():
    job_dir.mkdir()


# Hyperparams to try
lr_list = [1e-3] # [1e-3, 1e-4]
encoder_fc_dims_list = [[8]]
hidden_dims_list = [2]
odefunc_fc_dims_list = [[4]] # [[4], [8]]
decoder_fc_dims_list = [[8]]
window_length_list = [128] # [64, 96, 128]
batch_size_list = [1] # [1, 5, 10]
num_epochs_list = [256]
rtol_list = [1e-3]
atol_list = [1e-5]

# For each hyperparam combination, create an sbatch file to run
with open("run_all.sh", "w") as allf:

    for lr in lr_list:
        for encoder_fc_dims in encoder_fc_dims_list:
            for hidden_dims in hidden_dims_list:
                for odefunc_fc_dims in odefunc_fc_dims_list:
                    for decoder_fc_dims in decoder_fc_dims_list:
                        for window_length in window_length_list:
                            for batch_size in batch_size_list:
                                for num_epochs in num_epochs_list:
                                    for rtol in rtol_list:
                                        for atol in atol_list:

                                            job = (
                                                f"lr{lr:.1e}"
                                                + f"_enc{encoder_fc_dims}"
                                                + f"_hidden{hidden_dims}"
                                                + f"_ode{odefunc_fc_dims}"
                                                + f"_dec{decoder_fc_dims}"
                                                + f"_window{window_length}"
                                                + f"_batch{batch_size}"
                                                + f"_epochs{num_epochs}"
                                                + f"_rtol{rtol}"
                                                + f"_atol{atol}"
                                                )
                                            
                                            job_file = job_dir / f"{job}.job"

                                            with open(job_file, "w") as f:
                                                f.write(f"#!/bin/bash\n")
                                                f.write(f"#SBATCH -N 1\n")
                                                f.write(f"#SBATCH -n 1\n")
                                                f.write(f"#SBATCH --gres={gres}\n")
                                                f.write(f"#SBATCH --qos={QOS}\n")
                                                f.write(f"#SBATCH -p {partition}\n")
                                                f.write(f"#SBATCH --cpus-per-task={CPU}\n")
                                                f.write(f"#SBATCH --mem={RAM}\n")
                                                f.write(f"#SBATCH --job-name={job}\n")
                                                f.write(f"#SBATCH --output=%x.out\n")
                                                f.write(f"cd {root}\n")

                                                # Need to separate items in lists with spaces to pass them as args
                                                encoder_fc_dims_arg = " ".join(map(str, encoder_fc_dims))
                                                odefunc_fc_dims_arg = " ".join(map(str, odefunc_fc_dims))
                                                decoder_fc_dims_arg = " ".join(map(str, decoder_fc_dims))

                                                f.write(
                                                    f"python3 train.py "
                                                    + f"--lr={lr} "
                                                    + f"--encoder_fc_dims {encoder_fc_dims_arg} "
                                                    + f"--hidden_dims={hidden_dims} "
                                                    + f"--odefunc_fc_dims {odefunc_fc_dims_arg} "
                                                    + f"--decoder_fc_dims {decoder_fc_dims_arg} "
                                                    + f"--window_length={window_length} "
                                                    + f"--batch_size={batch_size} "
                                                    + f"--num_epochs={num_epochs} "
                                                    + f"--rtol={rtol} "
                                                    + f"--atol={atol}\n"
                                                )

                                            allf.write(f"sbatch {job_file}\n")
