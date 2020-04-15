#!/usr/bin/env python3
import sys
import time
import boto3
import argparse

CONDA_ENV = "tensorflow_p36"
USERNAME = "ubuntu"
SEQDESIGN_FOLDER = "AutoregressiveProteinModels"
AWS_REGION = 'us-west-2'
POWEROFF_TIME = 5  # number of minutes to wait after completion before terminating the instance

parser = argparse.ArgumentParser(description="Calculate the log probability of mutated sequences.")
parser.add_argument('script', type=str, nargs='+', default=[], help="Script(s) to schedule on new instances")
parser.add_argument("--instance-type", type=str, default='p2.xlarge', metavar='TYPE',
                    help="AWS instance type (e.g. p2.xlarge)")
parser.add_argument("--split-lines", action='store_true', help="Run every line in a separate instance")
parser.add_argument("--dry-run", action='store_true', help="Perform a dry run")
parser.add_argument("--run-version", type=str, default='v3', metavar='V',
                    help="Current run version (e.g. v2, v3, etc.).")
args = parser.parse_args()

home_path = f"/home/{USERNAME}"
seqdesign_path = f"{home_path}/{SEQDESIGN_FOLDER}"
env_bin_path = f"{home_path}/anaconda3/envs/{CONDA_ENV}/bin"
userdata_template = f"""#!/bin/bash
git config --system credential.helper '!aws codecommit credential-helper $@'
git config --system credential.UseHttpPath true
git clone https://git-codecommit.us-west-2.amazonaws.com/v1/repos/SeqDesignAR {seqdesign_path}
chown -R {USERNAME}:{USERNAME} {seqdesign_path}
su {USERNAME} -c '{env_bin_path}/pip install gitpython'
cd {seqdesign_path}
echo '#!/bin/bash
source activate {CONDA_ENV}
cd {seqdesign_path}
EXIT_STATUS=0
{{run_strings}}
if [ $EXIT_STATUS -ne 0 ]; then
    echo "Error detected. Syncing all logs to S3."
    aws s3 sync {seqdesign_path}/sess/ s3://markslab-private/autoregressive/{args.run_version}/sess/_failed_jobs/
fi
echo "Shutting down in {POWEROFF_TIME} minutes, press Ctrl-C to interrupt."
sleep {POWEROFF_TIME * 60} && sudo poweroff
' > run.sh
chown {USERNAME}:{USERNAME} run.sh
chmod +x run.sh
su - {USERNAME} -c "cd {seqdesign_path}
tmux new-session -s train -d -n 'train' 'bash'
tmux pipe-pane -o 'cat >> {seqdesign_path}/sess/tmux-output.#h.txt'
tmux send -t train.0 './run.sh' ENTER
"
"""

if __name__ == "__main__":
    sys.path.append(seqdesign_path)
    import aws_utils
    aws_utils.aws_s3_sync(
        local_folder=f"{seqdesign_path}/scheduling/aws_train/", s3_folder="scheduling/aws_train/", destination='s3'
    )

    if args.script is None:
        names = ["test_scheduler"]
        run_strings = [
            "python run_autoregressive_fr.py --dataset test_BLAT_ECOLX_1 --channels 8 "
            "--r-seed 11 --num-iterations 102 --snapshot-interval 50"
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
            f"{line.strip()} || EXIT_STATUS=$?"
            for line in run_string.splitlines(keepends=False)
            if line.strip()
        ])
        userdata = userdata_template.format(run_strings=run_string)
        try:
            response = ec2.run_instances(
                LaunchTemplate={"LaunchTemplateName": "SeqDesignTrain"},
                InstanceType=args.instance_type,
                UserData=userdata,
                TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": name}]}],
                InstanceInitiatedShutdownBehavior="terminate",
                MinCount=1,
                MaxCount=1,
                DryRun=args.dry_run,
            )
        except Exception as e:
            print(e)
        print("Launched.")
    print("Done.")
