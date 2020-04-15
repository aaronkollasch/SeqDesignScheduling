#!/usr/bin/env python3
import sys
import time
import boto3
import argparse

CONDA_ENV = "tensorflow_p36"
USERNAME = "ubuntu"
SEQDESIGN_FOLDER = "SeqDesign"
TRAIN_EXE = "run_autoregressive_fr"
AWS_REGION = 'us-west-2'
POWEROFF_TIME = 5  # number of minutes to wait after completion before terminating the instance

home_path = f"/home/{USERNAME}"
seqdesign_path = f"{home_path}/{SEQDESIGN_FOLDER}"
seqdesign_run_path = f"{seqdesign_path}/run"
env_bin_path = f"{home_path}/anaconda3/envs/{CONDA_ENV}/bin"
userdata_template = f"""#!/bin/bash
su {USERNAME} -c '
git clone -b v3 https://github.com/debbiemarkslab/SeqDesign.git {seqdesign_path}
{env_bin_path}/pip install gitpython
{env_bin_path}/python {seqdesign_path}/setup.py install
mkdir -p {seqdesign_run_path}
'
cd {seqdesign_run_path}
echo '#!/bin/bash
source activate {CONDA_ENV}
cd {seqdesign_run_path}
EXIT_STATUS=0
{{run_strings}}
if [ $EXIT_STATUS -ne 0 ]; then
    echo "Error detected. Syncing all logs to S3."
    aws s3 sync {seqdesign_run_path}/sess/ {{s3_path}}/{{run_version}}/sess/_failed_jobs/
fi
echo "Shutting down in {POWEROFF_TIME} minutes, press Ctrl-C to interrupt."
sleep {POWEROFF_TIME * 60} && sudo poweroff
' > run.sh
chown {USERNAME}:{USERNAME} run.sh
chmod +x run.sh
su - {USERNAME} -c "cd {seqdesign_run_path}
tmux new-session -s train -d -n 'train' 'bash'
tmux pipe-pane -o 'cat >> {seqdesign_run_path}/sess/tmux-output.#h.txt'
tmux send -t train.0 './run.sh' ENTER
"
"""

if __name__ == "__main__":
    sys.path.append(seqdesign_path)
    import aws_utils

    parser = argparse.ArgumentParser(description="Calculate the log probability of mutated sequences.")
    parser.add_argument('script', type=str, nargs='+', default=[], help="Script(s) to schedule on new instances")
    parser.add_argument("--instance-type", type=str, default='p2.xlarge', metavar='TYPE',
                        help="AWS instance type (e.g. p2.xlarge)")
    parser.add_argument("--split-lines", action='store_true', help="Run every line in a separate instance")
    parser.add_argument("--dry-run", action='store_true', help="Perform a dry run")
    parser.add_argument("--s3-path", type=str, default='s3://markslab-private/seqdesign',
                        help="Base s3:// SeqDesign path")
    parser.add_argument("--run-version", type=str, default='v3', metavar='V',
                        help="Current run version (e.g. v2, v3, etc.).")
    args = parser.parse_args()

    aws_util = aws_utils.AWSUtility(s3_base_path=args.s3_path)
    aws_util.s3_sync(
        local_folder=f"{home_path}/SeqDesignScheduling/aws_train/",
        s3_folder="scheduling/aws_train/",
        destination='s3',
        args=("--exclude", "'*.py'"),
    )

    if args.script is None:
        names = ["test_scheduler"]
        run_strings = [
            "--dataset test_BLAT_ECOLX_1 --channels 8 --r-seed 11 --num-iterations 102 --snapshot-interval 50 ",
        ]
        print("Usage: aws_schedule_train.py [script1] [script2] ...")
        print("Running test in 5 seconds (Press Ctrl-C to cancel).")
        time.sleep(5)
    else:
        names = args.script
        run_strings = []
        for fname in names:
            with open(fname) as f:
                run_strings.append(f.read().strip())
        names = [name.split('/')[-1] for name in names]
        names = [name[:name.rfind('.')] for name in names]

    if args.split_lines:
        new_names, new_strings = [], []
        for name, run_string in zip(names, run_strings):
            for i, line in enumerate(run_string.splitlines(keepends=False)):
                if line.lstrip().startswith('#') or not line.strip():
                    continue
                new_names.append(f'{name}_{i}')
                new_strings.append(line)
        names, run_strings = new_names, new_strings

    ec2 = boto3.client('ec2', region_name=AWS_REGION)
    for name, run_string in zip(names, run_strings):
        print(f"Launching instance {name} with commands:")
        print(run_string)
        run_string = '\n'.join([
            f"{TRAIN_EXE} {line.strip()} --s3-path {args.s3_path} || EXIT_STATUS=$?"
            for line in run_string.splitlines(keepends=False)
            if line.strip()
        ])
        userdata = userdata_template.format(
            run_strings=run_string,
            s3_path=args.s3_path,
            run_version=args.run_version
        )
        try:
            response = ec2.run_instances(
                LaunchTemplate={"LaunchTemplateName": "SeqDesignTrain"},
                InstanceType=args.instance_type,
                UserData=userdata,
                TagSpecifications=[
                    {"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": name}]},
                    {"ResourceType": "volume", "Tags": [{"Key": "Name", "Value": name}]},
                ],
                InstanceInitiatedShutdownBehavior="terminate",
                MinCount=1,
                MaxCount=1,
                DryRun=args.dry_run,
            )
        except Exception as e:
            print(e)
        print("Launched.")
    print("Done.")
