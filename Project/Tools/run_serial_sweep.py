"""
Run serial sweep of runs listed in master_runs.csv, ensuring the same max steps per run.

Usage example:
  python Tools/run_serial_sweep.py --master results/master_runs.csv \
    --config C:/soqqle/ml-agents/config/ppo/PushBlock.yaml \
    --outdir results --max-steps 500000 --env EDITOR

The script will:
 - Read the master CSV produced by experiment_harness.py
 - For each run, create a temporary copy of the trainer config with `max_steps` set to --max-steps
 - Execute `mlagents-learn` with that temp config and the run's run_id
 - Record start/end times, exit code, and tail of stdout/stderr into `master_runs_with_results.csv`

Note: This script runs `mlagents-learn` on the local machine. Ensure `mlagents-learn` is on PATH.
"""
import argparse
import csv
import os
import shutil
import subprocess
import tempfile
import time
import datetime
import yaml


def load_master(master_csv):
    rows = []
    with open(master_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def write_temp_config(base_config_path, out_path, max_steps):
    with open(base_config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # The ML-Agents trainer config typically has a top-level "behaviors" mapping.
    if 'behaviors' in cfg:
        for bname, bconf in cfg['behaviors'].items():
            # Set max_steps if present or create it
            bconf['max_steps'] = int(max_steps)
    else:
        # Try to set at top-level for backward compatibility
        cfg['max_steps'] = int(max_steps)

    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--master', required=True, help='Path to master_runs.csv')
    p.add_argument('--config', required=True, help='Base trainer YAML config')
    p.add_argument('--outdir', default='results', help='Output dir for results CSV')
    p.add_argument('--max-steps', type=int, required=True, help='Max steps per run (same for all)')
    p.add_argument('--env', default='', help='Path to Unity env or EDITOR for Editor')
    p.add_argument('--base-port', default='5004', help='Base port for communicator')
    p.add_argument('--cmd-template', default=None, help='Optional command template')
    p.add_argument('--sleep-between', type=int, default=0, help='Seconds to sleep between runs')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_master(args.master)

    results = []

    for r in rows:
        run_id = r['run_id']
        print(f"Preparing run {run_id}")
        # Create a temp config with max_steps set
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tf:
            temp_config_path = tf.name
        write_temp_config(args.config, temp_config_path, args.max_steps)

        if args.cmd_template:
            cmd = args.cmd_template.format(config=temp_config_path, run_id=run_id, env=args.env, base_port=args.base_port,
                                          mass=r.get('mass',''), size=r.get('size',''), friction=r.get('friction',''))
        else:
            cmd = f'mlagents-learn "{temp_config_path}" --run-id={run_id} --base-port={args.base_port} --env="{args.env}" --train'

        print(f"Starting: {cmd}")
        start = datetime.datetime.utcnow().isoformat()
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            exit_code = proc.returncode
            stdout = proc.stdout[-2000:]
            stderr = proc.stderr[-2000:]
        except Exception as e:
            exit_code = -1
            stdout = ''
            stderr = str(e)
        end = datetime.datetime.utcnow().isoformat()

        results.append({
            'run_id': run_id,
            'mass': r.get('mass',''),
            'size': r.get('size',''),
            'friction': r.get('friction',''),
            'start_time': start,
            'end_time': end,
            'exit_code': exit_code,
            'stdout_tail': stdout.replace('\n','\\n'),
            'stderr_tail': stderr.replace('\n','\\n'),
        })

        # Clean temp config
        try:
            os.remove(temp_config_path)
        except Exception:
            pass

        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    out_master_with_results = os.path.join(args.outdir, 'master_runs_with_results.csv')
    with open(out_master_with_results, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['run_id','mass','size','friction','start_time','end_time','exit_code','stdout_tail','stderr_tail']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rr in results:
            writer.writerow(rr)

    print(f"Finished. Results written to {out_master_with_results}")


if __name__ == '__main__':
    main()
