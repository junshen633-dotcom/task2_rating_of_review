#!/usr/bin/env python3
"""Monitor one authorized SLURM job during an interactive session."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--work', type=Path, required=True)
p.add_argument('--job', required=True)
p.add_argument('--interval', type=int, default=30)
p.add_argument('--runs', nargs='+', default=['E4_long8_seed43'])
p.add_argument('--log-prefix', default='extended')
a = p.parse_args()
last_state = last_error = None
last_epochs = {}
terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED'}
while True:
    query = subprocess.run(['sacct', '-n', '-P', '-j', a.job, '--format=JobIDRaw,State,ExitCode,Elapsed'], capture_output=True, text=True)
    rows = [r.split('|') for r in query.stdout.splitlines() if r.split('|')[0] == a.job]
    if rows:
        row = rows[0]
        state = row[1].split()[0].split('+')[0]
        if state != last_state:
            print(json.dumps({'job': a.job, 'state': state, 'exit': row[2], 'elapsed': row[3]}), flush=True)
            last_state = state
    else:
        state = None
    for name in a.runs:
        progress = a.work/'runs'/name/'training_progress.json'
        if progress.exists():
            report = json.loads(progress.read_text())
            if report['epochs']:
                epoch = report['epochs'][-1]
                if epoch['epoch'] != last_epochs.get(name):
                    print(json.dumps({'run': name, 'progress': epoch, 'best_observed_epoch': report.get('best_epoch'), 'best_latent_epoch': report.get('best_latent_epoch')}), flush=True)
                    last_epochs[name] = epoch['epoch']
    error = a.work/'logs'/(a.log_prefix+'-'+a.job+'.err')
    if error.exists():
        value = error.read_text(errors='replace')
        if value and value != last_error:
            print(json.dumps({'stderr_tail': value[-5000:]}), flush=True)
            last_error = value
    if state in terminal:
        print(json.dumps({'monitor_finished': True, 'state': state}), flush=True)
        raise SystemExit(0 if state == 'COMPLETED' else 1)
    time.sleep(max(5, a.interval))
