#!/usr/bin/env python3
"""Evaluate a cal-frozen recommendation on the previously used eval benchmark.

Never select models or weights from this result. Not a new blind evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from ensemble_predictions import equal_weight, score, sorted_predictions
from postprocess import digest, read_predictions, validate_transition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    a = p.parse_args()
    selection_path = a.bundle/'selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if selection['selection_split'] != 'cal':
        raise ValueError('Recommendation must be frozen using cal only')
    destination = a.bundle/'evaluation_selected.json'
    if destination.exists():
        raise FileExistsError('Evaluation already reported; do not repeat selection on eval')
    metadata_path = a.work/'prepared/metadata.json'
    metadata = json.loads(metadata_path.read_text())
    transition = validate_transition(metadata['transition'])
    metadata_hash = digest(metadata_path)
    baseline_path = a.work/'final_E4_seed43/predictions_eval.csv'
    baseline_metrics = json.loads((a.work/'final_E4_seed43/metrics.json').read_text())
    if baseline_metrics['metadata_sha256'] != metadata_hash:
        raise ValueError('Baseline evaluation metadata mismatch')
    baseline = sorted_predictions(read_predictions(baseline_path, labelled=True))
    members, provenance = [], {}
    for name in selection['selected_models']:
        if name == 'E4_seed43':
            members.append(baseline)
            provenance[name] = {'predictions_sha256': digest(baseline_path), 'reused_baseline': True}
            continue
        folder = a.work/'runs'/name
        manifest = json.loads((folder/'metrics.json').read_text())
        if manifest['metadata_sha256'] != metadata_hash or manifest['smoke']:
            raise ValueError('Member metadata mismatch or smoke')
        checkpoint = Path(manifest['checkpoint'])
        output = a.work/'extended_selected_eval'/name
        if not (output/'metrics.json').exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name('train.py')),
                            '--data-dir', str(a.work/'prepared'), '--output-dir', str(output),
                            '--predict-only', '--checkpoint', str(checkpoint), '--evaluate-final',
                            '--device', 'cuda', '--amp', '--seed', str(manifest['seed']),
                            '--workers', '2', '--threads', '4'], check=True)
        eval_manifest = json.loads((output/'metrics.json').read_text())
        if eval_manifest['metadata_sha256'] != metadata_hash or eval_manifest['checkpoint'] != str(checkpoint.resolve()):
            raise ValueError('Existing prediction provenance mismatch')
        path = output/'predictions_eval.csv'
        members.append(sorted_predictions(read_predictions(path, labelled=True)))
        provenance[name] = {'checkpoint_sha256': digest(checkpoint), 'predictions_sha256': digest(path)}
    predictions = equal_weight(members)
    report = {'scope': 'Previously used eval benchmark; not new blind data; no selection or fitting',
              'selection_sha256': digest(selection_path), 'selected_name': selection['selected_name'],
              'selected_models': selection['selected_models'], 'metadata_sha256': metadata_hash,
              'members': provenance, 'baseline': score(baseline, transition, baseline),
              'selected': score(predictions, transition, baseline)}
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    readme = a.bundle/'README_CN.md'
    with readme.open('a', encoding='utf-8') as stream:
        stream.write('\n冻结候选的既有 eval 基准复核（不是新盲测，不用于再选模型）：\n\n')
        stream.write('| 模型 | 观察 RMSE | 条件修正潜在 MSE |\n|---|---:|---:|\n')
        for name, values in [('原 E4_seed43', report['baseline']), (selection['selected_name'], report['selected'])]:
            observed = values['observed_against_observed_rating']['rmse']
            risk = values['conditional_corrected_latent_risk']['estimated_latent_mse']
            stream.write(f'| {name} | {observed:.8f} | {risk:.8f} |\n')
    inventory = [f'{digest(path)}  {path.relative_to(a.bundle).as_posix()}'
                 for path in sorted(a.bundle.rglob('*')) if path.is_file() and path.name != 'SHA256SUMS']
    (a.bundle/'SHA256SUMS').write_text('\n'.join(inventory)+'\n', encoding='utf-8')
    print(json.dumps({'final_eval_frozen_candidate': selection['selected_name'],
                      'observed_rmse': report['selected']['observed_against_observed_rating']['rmse'],
                      'conditional_corrected_latent_risk': report['selected']['conditional_corrected_latent_risk']}, indent=2))

if __name__ == '__main__':
    main()
