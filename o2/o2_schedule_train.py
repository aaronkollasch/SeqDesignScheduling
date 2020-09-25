#!/n/groups/marks/users/aaron/anaconda3/envs/tensorflow_gpuenv/bin/python
import sys
import os
import time
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Calculate the log probability of mutated sequences.")
parser.add_argument('params', type=str, nargs='+', default=[], help="File(s) with params for each run.")
parser.add_argument("--gpu-type", type=str, default=None, metavar='TYPE',
                    help="GPU type (leave blank for no preference)")
parser.add_argument("--dry-run", action='store_true', help="Perform a dry run")
parser.add_argument("--s3-path", type=str, default='s3://markslab-private/seqdesign',
                    help="Base s3:// SeqDesign path")
parser.add_argument("--s3-project", type=str, default='v3', metavar='V',
                    help="Project name (subfolder of s3-path).")
args = parser.parse_args()

seqdesign_path = f"/n/groups/marks/projects/seqdesign/{args.s3_project}"

env_bin_path = f"/n/groups/marks/users/aaron/anaconda3/envs/tensorflow_gpuenv/bin"
sbatch_template = f"""#!/bin/bash
#SBATCH -c 2
#SBATCH --mem 40G
#SBATCH -t 59:59:00
#SBATCH -p gpu
#SBATCH --gres=gpu{':' + args.gpu_type if args.gpu_type else ''}:1
#SBATCH -o {seqdesign_path}/slurm/slurm_%j_{{name}}.out

cd {seqdesign_path}
{env_bin_path}/python seqdesign/scripts/run_autoregressive_fr.py {{params}}
"""

if args.params is None:
    names = ["test_scheduler"]
    param_strings = [
        "--dataset test_BLAT_ECOLX_1 --channels 8 --r-seed 11 --num-iterations 102 --snapshot-interval 50",
        "--dataset test_BLAT_ECOLX_1 --channels 8 --r-seed 22 --num-iterations 102 --snapshot-interval 50",
    ]
    print("Usage: o2_schedule_train.py [param_file1] [param_file2] ...")
    print("Running test in 5 seconds (Press Ctrl-C to cancel).")
    time.sleep(5)
else:
    names, param_strings = [], []
    for fname in args.params:
        with open(fname) as f:
            run_string = f.read().strip()
            for i, line in enumerate(run_string.splitlines(keepends=False)):
                if line.lstrip().startswith('#') or not line.strip():
                    continue
                name = fname.split('/')[-1]
                name = name[:name.rfind('.')]
                line = f"{line.strip()} --s3-path {args.s3_path}"
                names.append(f'{name}_{i}')
                param_strings.append(line)

os.makedirs(seqdesign_path, exist_ok=True)
os.makedirs('sbatch', exist_ok=True)
for name, param_string in zip(names, param_strings):
    print(f"sbatch run {name} with params {param_string}")

    sbatch_fname = f'sbatch/{name}.sh'
    with open(sbatch_fname, 'w') as job_file:
        job_file.write(sbatch_template.format(name=name, params=param_string))

    if not args.dry_run:
        pipes = subprocess.Popen(['sbatch', sbatch_fname], stdout=subprocess.PIPE, encoding='UTF-8')
        std_out, std_err = pipes.communicate()
        with open('sbatch/jobs.txt', 'a') as f:
            f.write(f"{sbatch_fname}\t{std_out.strip()}\n")
        print(std_out.strip())
