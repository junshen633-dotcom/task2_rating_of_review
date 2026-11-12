"""Choose objective using cal only, then run the preplanned OOF-feature ablation."""
import json
from pathlib import Path
import subprocess
import sys

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work', type=Path, required=True)
root = parser.parse_args().work
choices=[]
for name in ('E1','E2','E3'):
    m=json.loads((root/'runs'/name/'metrics.json').read_text())
    if m['smoke'] or m['evaluation_split_used_for_selection']:
        raise ValueError('Invalid source experiment')
    choices.append((m['best_cal_rmse'],name,m['architecture']['objective']))
score,name,objective=min(choices)
selection={'selection_split':'cal','source':name,'objective':objective,'source_cal_rmse':score,
           'candidate_metrics':choices,'change':'add OOF entity and dense metadata; other training arguments unchanged'}
(root/'E4_selection.json').write_text(json.dumps(selection,indent=2))
print(json.dumps(selection),flush=True)
subprocess.run([sys.executable,'train.py','--data-dir',str(root/'prepared'),
    '--output-dir',str(root/'runs'/'E4'),'--objective',objective,'--metadata',
    '--device','cuda','--amp','--max-epochs','3','--patience','2',
    '--batch-size','128','--workers','2','--threads','4'],check=True)
