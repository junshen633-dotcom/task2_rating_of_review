#!/usr/bin/env python3
"""Freeze text-ablation candidates on cal; then audit on the existing eval split.

No leaderboard weights are fitted. Conditional corrected MSE is not measured
clean-label error. Evaluation never changes the frozen recommendation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from ensemble_predictions import (UNCERTAINTY, assert_aligned, corrected_losses,
                                  equal_weight, paired_difference, score,
                                  sorted_predictions)
from postprocess import (align_to_sample, digest, read_predictions, read_sample,
                         save_submission, validate_transition)

INCUMBENT = ['E3', 'E4', 'E3_seed43', 'E4_seed43']
MATCHED = 'E4_long8_seed43_latent'
VARIANTS = [
    ('T1_separate_mean_seed43', 'separate', 'mean'),
    ('T2_joint_meanmax_seed43', 'joint', 'mean_max'),
    ('T3_separate_meanmax_seed43', 'separate', 'mean_max'),
]
COMMON_ARCHITECTURE = ('objective', 'backbone', 'vocab_size', 'max_tokens',
                       'embedding_dim', 'cnn_channels', 'hidden_dim', 'dropout',
                       'kernels', 'dense_dim', 'label_mean')
COMMON_ARGS = ('batch_size', 'weight_decay', 'workers', 'threads', 'shuffle_buffer', 'amp')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def check_manifest(record, metadata_hash, name):
    if record.get('metadata_sha256') != metadata_hash:
        raise ValueError(f'Metadata mismatch: {name}')
    if record.get('smoke') is not False or record.get('evaluation_split_used_for_selection') is not False:
        raise ValueError(f'Formal cal-only provenance is missing: {name}')


def load_member(work, name, split, metadata_hash):
    folder = work/'runs'/name
    manifest = read_json(folder/'metrics.json')
    check_manifest(manifest, metadata_hash, name)
    path = folder/f'predictions_{split}.csv'
    predictions = sorted_predictions(read_predictions(path, labelled=split != 'test'))
    return predictions, manifest, {'metrics_sha256': digest(folder/'metrics.json'),
                                  'predictions_sha256': digest(path),
                                  'checkpoint': manifest.get('checkpoint'),
                                  'best_epoch': manifest.get('best_epoch')}


def parameter_count(architecture):
    """Exact trainable count for shared-weights HashTextCNN + the existing MLP."""
    if architecture['backbone'] != 'hashcnn':
        raise ValueError('This ablation requires HashTextCNN')
    vocab, dim = architecture['vocab_size'], architecture['embedding_dim']
    channels, kernels = architecture['cnn_channels'], architecture.get('kernels', [3, 5])
    width = channels*len(kernels)
    width *= 2 if architecture.get('pooling', 'mean') == 'mean_max' else 1
    width *= 2 if architecture.get('text_layout', 'joint') == 'separate' else 1
    dense, hidden = architecture.get('dense_dim', 0), architecture['hidden_dim']
    outputs = 1 if architecture['objective'] == 'mse' else 5
    return {'text_representation_dim': width,
            'trainable_parameters': vocab*dim + sum(channels*dim*k+channels for k in kernels)
            + (width+dense)*hidden+hidden + hidden*outputs+outputs,
            'basis': 'Calculated from recorded shared-encoder architecture; includes changed MLP input width'}


def training_contract(work, name, prediction_manifest, metadata_hash,
                      expected_layout, expected_pooling, baseline=None, matched=False):
    source_name = 'E4_long8_seed43' if matched else name
    path = work/'runs'/source_name/'metrics.json'
    training = read_json(path)
    check_manifest(training, metadata_hash, source_name)
    if training.get('seed') != 43 or prediction_manifest.get('seed') != 43:
        raise ValueError(f'Ablation seed must be 43: {name}')
    architecture = training['architecture']
    if architecture != prediction_manifest['architecture']:
        raise ValueError(f'Prediction architecture differs from training: {name}')
    if architecture.get('text_layout', 'joint') != expected_layout or architecture.get('pooling', 'mean') != expected_pooling:
        raise ValueError(f'Wrong representation for {name}')
    if architecture['objective'] != 'forward' or architecture.get('dense_dim', 0) != 12:
        raise ValueError('All four ablation cells require forward loss and 12 dense metadata features')
    prefix = training['epochs'][:4]
    if len(prefix) != 4 or [e['epoch'] for e in prefix] != [1, 2, 3, 4]:
        raise ValueError(f'Four-epoch prefix is missing: {name}')
    if not np.allclose([e['learning_rate'] for e in prefix], [.001, .001, .001, .0005], rtol=0, atol=1e-12):
        raise ValueError(f'Unexpected learning-rate prefix: {name}')
    if not matched and (len(training['epochs']) != 4 or training['args']['max_epochs'] != 4):
        raise ValueError(f'New ablations must complete exactly four epochs: {name}')
    best = min(prefix, key=lambda e: e['calibration']['corrected_latent_risk']['estimated_latent_mse'])
    if training.get('best_latent_epoch') != best['epoch'] or prediction_manifest.get('best_epoch') != best['epoch']:
        raise ValueError(f'Predictions are not from the first-four-epoch latent optimum: {name}')
    if matched and best['epoch'] != 4:
        raise ValueError('Reused historical control must be the verified epoch-4 checkpoint')
    if prediction_manifest['checkpoint'] != training['best_latent_checkpoint']:
        raise ValueError(f'Wrong checkpoint exported: {name}')
    if baseline is not None:
        for key in COMMON_ARCHITECTURE:
            if architecture[key] != baseline['architecture'][key]:
                raise ValueError(f'Architecture confound {key}: {name}')
        for key in COMMON_ARGS:
            if training['args'][key] != baseline['args'][key]:
                raise ValueError(f'Training-setting confound {key}: {name}')
        if [e['train_records'] for e in prefix] != [e['train_records'] for e in baseline['epochs'][:4]]:
            raise ValueError('Ablation training coverage differs')
    counts = parameter_count(architecture)
    recorded = training.get('parameter_count', training.get('parameter_counts'))
    if isinstance(recorded, int) and recorded != counts['trainable_parameters']:
        raise ValueError(f'Recorded parameter count differs from architecture: {name}')
    return training, {'name': name, 'layout': expected_layout, 'pooling': expected_pooling,
                      'seed': 43, 'best_epoch': best['epoch'], 'completed_epochs_for_comparison': 4,
                      'best_at_budget_boundary': best['epoch'] == 4,
                      'reused_first_four_epoch_control': matched,
                      'architecture': architecture, 'training_args': training['args'],
                      'epochs': prefix, **counts, 'recorded_parameter_count': recorded,
                      'metrics_sha256': digest(path)}


def risk_value(report):
    risk = report['conditional_corrected_latent_risk']
    if not risk.get('available') or not np.isfinite(risk['estimated_latent_mse']):
        raise ValueError('Finite corrected risk required for cal selection')
    return float(risk['estimated_latent_mse'])


def choose_candidates(incumbent, new_scores, five_score):
    """Stable ties retain the incumbent, with no numeric weight search."""
    best_new = min(new_scores, key=lambda name: risk_value(new_scores[name]))
    choices = [('current_four', risk_value(incumbent)),
               (best_new, risk_value(new_scores[best_new])),
               ('five_model', risk_value(five_score))]
    return best_new, min(choices, key=lambda item: item[1])[0]


def refresh_checksums(out):
    lines = [f'{digest(path)}  {path.relative_to(out).as_posix()}'
             for path in sorted(out.rglob('*')) if path.is_file() and path.name != 'SHA256SUMS']
    (out/'SHA256SUMS').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def collect(args):
    work, out = args.work, args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing nonempty output: {out}')
    metadata_path = work/'prepared/metadata.json'
    metadata_hash = digest(metadata_path)
    transition = validate_transition(read_json(metadata_path)['transition'])
    sample_ids = read_sample(args.sample)
    if len(sample_ids) != 10000:
        raise ValueError('Expected 10000 unique sample IDs')
    all_names = INCUMBENT+[MATCHED]+[name+'_latent' for name, _, _ in VARIANTS]
    predictions, tests, manifests, provenance = {}, {}, {}, {}
    for name in all_names:
        predictions[name], manifests[name], provenance[name] = load_member(work, name, 'cal', metadata_hash)
        tests[name], _, test_source = load_member(work, name, 'test', metadata_hash)
        provenance[name]['test_predictions_sha256'] = test_source['predictions_sha256']
        align_to_sample(tests[name].ids, tests[name].latent, sample_ids)
    incumbent = equal_weight([predictions[name] for name in INCUMBENT])
    matched = predictions[MATCHED]
    control_training, control = training_contract(work, MATCHED, manifests[MATCHED], metadata_hash,
                                                  'joint', 'mean', matched=True)
    cell_records = [control]
    for name, layout, pooling in VARIANTS:
        _, record = training_contract(work, name, manifests[name+'_latent'], metadata_hash,
                                      layout, pooling, baseline=control_training)
        cell_records.append(record)
    incumbent_score = score(incumbent, transition, incumbent)
    cell_scores = {name: score(predictions[name], transition, incumbent)
                   for name in [MATCHED]+[n+'_latent' for n, _, _ in VARIANTS]}
    new_scores = {name+'_latent': cell_scores[name+'_latent'] for name, _, _ in VARIANTS}
    best_new = min(new_scores, key=lambda name: risk_value(new_scores[name]))
    five_members = INCUMBENT+[best_new]
    five = equal_weight([predictions[name] for name in five_members])
    five_score = score(five, transition, incumbent)
    confirmed_best, selected_name = choose_candidates(incumbent_score, new_scores, five_score)
    assert best_new == confirmed_best
    models = {'current_four': INCUMBENT, best_new: [best_new], 'five_model': five_members}
    scores = {'current_four': incumbent_score, best_new: new_scores[best_new], 'five_model': five_score}
    selected_models = models[selected_name]
    for cell in cell_records:
        name = MATCHED if cell['reused_first_four_epoch_control'] else cell['name']+'_latent'
        cell['scores_vs_current_four'] = cell_scores[name]
        cell['paired_corrected_mse_vs_matched_single'] = paired_difference(
            corrected_losses(predictions[name].latent, predictions[name].labels, transition),
            corrected_losses(matched.latent, matched.labels, transition))
    values = [risk_value(cell_scores[name]) for name in [MATCHED]+[n+'_latent' for n, _, _ in VARIANTS]]
    factorial = {'separation_effect_with_mean': values[1]-values[0],
                 'mean_max_effect_with_joint': values[2]-values[0],
                 'combined_effect': values[3]-values[0],
                 'difference_of_differences': values[3]-values[1]-values[2]+values[0],
                 'interpretation': 'Descriptive fixed-seed/four-epoch contrast; head width also changes, not a causal isolated operator estimate'}
    submissions = {'current_four_latent.csv': equal_weight([tests[n] for n in INCUMBENT]),
                   'matched_single_epoch4_latent.csv': tests[MATCHED],
                   'five_model_latent.csv': equal_weight([tests[n] for n in five_members])}
    for name, _, _ in VARIANTS:
        submissions[name+'_latent.csv'] = tests[name+'_latent']
    selected_test = equal_weight([tests[name] for name in selected_models])
    aligned = {filename: align_to_sample(p.ids, p.latent, sample_ids) for filename, p in submissions.items()}
    aligned['recommended.csv'] = align_to_sample(selected_test.ids, selected_test.latent, sample_ids)
    selection = {'selection_split': 'cal', 'evaluation_read': False, 'leaderboard_used_for_selection': False,
                 'selected_name': selected_name, 'selected_models': selected_models,
                 'selected_weights': [1/len(selected_models)]*len(selected_models),
                 'best_new_name': best_new, 'best_new_models': [best_new],
                 'selected_corrected_latent_mse': risk_value(scores[selected_name]),
                 'recommended_file': 'recommended.csv', 'current_four_models': INCUMBENT,
                 'current_four': incumbent_score, 'new_models': new_scores, 'five_model': five_score,
                 'five_model_members': five_members, 'five_model_weights': [0.2]*5,
                 'ablation_cells': cell_records, 'factorial_contrasts': factorial,
                 'policy': 'Choose best new model by cal corrected latent MSE. Then choose incumbent, best new, or direct five-member equal mean; ties keep incumbent. No weight search.',
                 'metadata_sha256': metadata_hash, 'sample_sha256': digest(args.sample),
                 'member_provenance': provenance, 'submission_files': list(aligned),
                 'uncertainty': UNCERTAINTY,
                 'parameter_warning': 'Same lexical token budget and shared convolution weights; MLP input width and parameter count change across representation schemes.',
                 'kaggle_submitted': False}
    out.mkdir(parents=True, exist_ok=True)
    for filename, values in aligned.items():
        save_submission(out/filename, sample_ids, values)
    write_json(out/'selection.json', selection)
    lines = ['# 文本表示对照实验', '', f'推荐提交：`recommended.csv`（{selected_name}）。', '',
             '同一seed43、前四轮预算和学习率、相同截断后的tokens、同一噪声矩阵。分开编码共享embedding和CNN，但分别池化；融合层参数量随表示维数增加。', '',
             '| 单模型表示 | 最佳轮次 | 参数量 | cal观察RMSE | 条件修正潜在MSE |',
             '|---|---:|---:|---:|---:|']
    for cell in cell_records:
        value = cell['scores_vs_current_four']
        lines.append(f"| {cell['layout']} / {cell['pooling']} | {cell['best_epoch']} | {cell['trainable_parameters']} | {value['observed_against_observed_rating']['rmse']:.8f} | {risk_value(value):.8f} |")
    lines += ['', '| 部署候选 | cal观察RMSE | 条件修正潜在MSE |', '|---|---:|---:|']
    for name, value in scores.items():
        lines.append(f"| {name} | {value['observed_against_observed_rating']['rmse']:.8f} | {risk_value(value):.8f} |")
    lines += ['', '单模型消融首先与同预算单模型比较；与四模型集成比较用于决定提交候选。不能把新单模型未超越集成理解为表示改动无效。',
              '五模型融合直接平均四个旧模型与选定新模型，每个权重20%；不是旧集成与新模型各50%。',
              '修正潜在MSE依赖固定类别条件噪声假设，不是实测干净标签误差，不转换为榜单RMSE。仅按cal选优；eval在冻结后报告且不再选模型。',
              '每份CSV均验证10000行、唯一ID、模板顺序及[1,5]范围。当前新文件没有Kaggle成绩。',
              '四轮为相同预算初筛，不代表所有架构已收敛；单seed结果尚不证明跨seed稳定。', '']
    (out/'README_CN.md').write_text('\n'.join(lines), encoding='utf-8')
    records = out/'training'; records.mkdir()
    for name in ['E4_long8_seed43']+[n for n, _, _ in VARIANTS]:
        shutil.copyfile(work/'runs'/name/'metrics.json', records/(name+'.json'))
    refresh_checksums(out)
    print(json.dumps({'selected': selected_name, 'models': selected_models,
                      'corrected_latent_mse': risk_value(scores[selected_name]),
                      'best_new': best_new, 'out': str(out)}, indent=2))
    return selection


def load_eval(work, name, metadata_hash):
    if name == 'E4_seed43':
        folder = work/'final_E4_seed43'
    elif name in INCUMBENT:
        folder = work/'extended_selected_eval'/name
    else:
        source = read_json(work/'runs'/name/'metrics.json')
        check_manifest(source, metadata_hash, name)
        checkpoint = Path(source['checkpoint'])
        folder = work/'text_selected_eval'/name
        if not (folder/'metrics.json').exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name('train.py')),
                            '--data-dir', str(work/'prepared'), '--output-dir', str(folder),
                            '--predict-only', '--checkpoint', str(checkpoint), '--evaluate-final',
                            '--device', 'cuda', '--amp', '--seed', str(source['seed']),
                            '--workers', '2', '--threads', '4'], check=True)
        if read_json(folder/'metrics.json')['checkpoint'] != str(checkpoint.resolve()):
            raise ValueError(f'Cached evaluation checkpoint mismatch: {name}')
    manifest = read_json(folder/'metrics.json')
    check_manifest(manifest, metadata_hash, name)
    path = folder/'predictions_eval.csv'
    return sorted_predictions(read_predictions(path, labelled=True)), {
        'metrics_sha256': digest(folder/'metrics.json'), 'predictions_sha256': digest(path),
        'checkpoint': manifest['checkpoint']}


def evaluate(args):
    work, out = args.work, args.out
    target = out/'evaluation.json'
    if target.exists():
        raise FileExistsError('Evaluation already reported; frozen selection cannot be re-tuned here')
    selection_path = out/'selection.json'
    selection = read_json(selection_path)
    if selection['selection_split'] != 'cal' or selection['evaluation_read']:
        raise ValueError('Expected a cal-only frozen selection')
    metadata_path = work/'prepared/metadata.json'
    metadata_hash = digest(metadata_path)
    if metadata_hash != selection['metadata_sha256']:
        raise ValueError('Metadata changed after selection')
    transition = validate_transition(read_json(metadata_path)['transition'])
    members, provenance = {}, {}
    needed = list(dict.fromkeys(INCUMBENT+selection['best_new_models']+selection['selected_models']))
    for name in needed:
        members[name], provenance[name] = load_eval(work, name, metadata_hash)
    incumbent = equal_weight([members[n] for n in INCUMBENT])
    best_new = equal_weight([members[n] for n in selection['best_new_models']])
    selected = equal_weight([members[n] for n in selection['selected_models']])
    report = {'scope': 'Previously used eval benchmark; not new blind data. No fitting or reselection.',
              'selection_sha256': digest(selection_path), 'metadata_sha256': metadata_hash,
              'selected_name': selection['selected_name'], 'selected_models': selection['selected_models'],
              'best_new_name': selection['best_new_name'], 'member_provenance': provenance,
              'current_four': score(incumbent, transition, incumbent),
              'best_new': score(best_new, transition, incumbent),
              'selected': score(selected, transition, incumbent)}
    write_json(target, report)
    with (out/'README_CN.md').open('a', encoding='utf-8') as stream:
        stream.write('\n冻结后的已有eval基准复核（不再选模型）：\n\n| 方案 | 观察RMSE | 条件修正潜在MSE |\n|---|---:|---:|\n')
        for name in ['current_four', 'best_new', 'selected']:
            value = report[name]
            stream.write(f"| {name} | {value['observed_against_observed_rating']['rmse']:.8f} | {risk_value(value):.8f} |\n")
    refresh_checksums(out)
    print(json.dumps({'selected': selection['selected_name'],
                      'eval_corrected_latent_mse': risk_value(report['selected']),
                      'baseline_eval_corrected_latent_mse': risk_value(report['current_four'])}, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['cal', 'eval'], required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sample', type=Path)
    args = p.parse_args()
    if args.stage == 'cal' and args.sample is None:
        p.error('--sample is required for cal collection')
    (collect if args.stage == 'cal' else evaluate)(args)

if __name__ == '__main__':
    main()
