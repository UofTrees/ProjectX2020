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
job_id_list = [
    "lr1.0e-03_enc[8, 16, 8]_hidden10_ode[4, 8, 8, 4]_dec[8, 16, 8]_window128_epochs256_rtol0.0001_atol1e-06",
    "lr1.0e-03_enc[8, 16, 8]_hidden10_ode[16, 32, 32, 16]_dec[8, 16, 8]_window128_epochs256_rtol0.0001_atol1e-06",
    "lr1.0e-04_enc[8, 16, 8]_hidden10_ode[4, 8, 8, 4]_dec[8, 16, 8]_window128_epochs256_rtol0.0001_atol1e-06",
    "lr1.0e-04_enc[8, 16, 8]_hidden10_ode[16, 32, 32, 16]_dec[8, 16, 8]_window128_epochs256_rtol0.0001_atol1e-06",
]


# For each hyperparam combination, create an sbatch file to run
with open("eval_all.sh", "w") as allf:

    for job_id in job_id_list:
        job_file = job_dir / f"{job_id}.job"
        job_out_file = job_dir / f"{job_id}.out"

        with open(job_file, "w") as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"#SBATCH -N 1\n")
            f.write(f"#SBATCH -n 1\n")
            f.write(f"#SBATCH --gres={gres}\n")
            f.write(f"#SBATCH --qos={QOS}\n")
            f.write(f"#SBATCH -p {partition}\n")
            f.write(f"#SBATCH --cpus-per-task={CPU}\n")
            f.write(f"#SBATCH --mem={RAM}\n")
            f.write(f"#SBATCH --job-name='{job_id}'\n")
            f.write(f"#SBATCH --output='{job_out_file}'\n")
            f.write(f"cd {root}\n")

            f.write(f"python3 eval.py --job_id='{job_id}'\n")

        allf.write(f"sbatch '{job_file}'\n")
