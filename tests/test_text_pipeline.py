"""Prespecified text-ablation selection and matched-budget provenance."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ensemble_predictions import equal_weight
from postprocess import Predictions
from run_text_ablation import (choose_candidates, parameter_count, refresh_checksums,
                               training_contract)


def risk(value):
    return {'conditional_corrected_latent_risk': {'available': True, 'estimated_latent_mse': value}}


def test_fixed_selection_preserves_incumbent_on_ties():
    best, selected = choose_candidates(risk(.2), {'a': risk(.3), 'b': risk(.25)}, risk(.2))
    assert best == 'b' and selected == 'current_four'
    assert choose_candidates(risk(.2), {'a': risk(.3), 'b': risk(.25)}, risk(.19))[1] == 'five_model'
    with pytest.raises(ValueError):
        choose_candidates(risk(.2), {'a': risk(float('nan'))}, risk(.2))


def test_direct_five_member_average_is_twenty_percent_new():
    def p(value):
        return Predictions(np.array([2,1]),np.array([value,value]),np.array([0,0]),
                           np.array([3.,3.]),np.array([value,value]))
    result = equal_weight([p(1.)]*4+[p(5.)])
    assert np.array_equal(result.ids, [1,2])
    assert np.allclose(result.latent, 1.8)
    assert not np.allclose(result.latent, 3.)


@pytest.mark.parametrize('layout,pooling,width', [('joint','mean',128),('separate','mean',256),
                                                ('joint','mean_max',256),('separate','mean_max',512)])
def test_recorded_parameter_formula_matches_model(layout, pooling, width):
    from model import ReviewModel
    import torch
    config = {'objective':'forward','backbone':'hashcnn','vocab_size':128,'embedding_dim':64,
              'cnn_channels':64,'hidden_dim':128,'dense_dim':12,'kernels':[3,5],
              'text_layout':layout,'pooling':pooling}
    model = ReviewModel(config, torch.eye(5))
    result = parameter_count(config)
    assert result['text_representation_dim'] == width
    assert result['trainable_parameters'] == sum(p.numel() for p in model.parameters())


def test_training_contract_rejects_wrong_epoch_and_schedule(tmp_path):
    config = {'objective':'forward','backbone':'hashcnn','vocab_size':128,'max_tokens':160,
              'embedding_dim':64,'cnn_channels':64,'hidden_dim':128,'dense_dim':12,
              'dropout':.2,'kernels':[3,5],'label_mean':3.8}
    epochs = [{'epoch':i, 'learning_rate': .001 if i<4 else .0005,
               'train_records':10, 'calibration':{'corrected_latent_risk':{'estimated_latent_mse':1/i}}}
              for i in range(1,5)]
    record = {'seed':43, 'metadata_sha256':'test', 'smoke':False,
              'evaluation_split_used_for_selection':False, 'architecture':config,
              'epochs':epochs, 'best_latent_epoch':4, 'best_latent_checkpoint':'checkpoint.pt',
              'args':{'max_epochs':4}}
    directory=tmp_path/'runs/E4_long8_seed43';directory.mkdir(parents=True)
    path=directory/'metrics.json';path.write_text(json.dumps(record))
    prediction={**record, 'best_epoch':4, 'checkpoint':'checkpoint.pt'}
    _, result=training_contract(tmp_path, 'E4_long8_seed43_latent', prediction,
                               'test','joint','mean',matched=True)
    assert result['best_epoch']==4 and result['reused_first_four_epoch_control']
    wrong={**prediction,'best_epoch':3}
    with pytest.raises(ValueError, match='optimum'):
        training_contract(tmp_path,'E4_long8_seed43_latent',wrong,'test','joint','mean',matched=True)
    record['epochs'][3]['learning_rate']=.001;path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='learning-rate'):
        training_contract(tmp_path,'E4_long8_seed43_latent',prediction,'test','joint','mean',matched=True)


def test_checksum_refresh_excludes_itself(tmp_path):
    (tmp_path/'a.txt').write_text('test')
    refresh_checksums(tmp_path)
    first=(tmp_path/'SHA256SUMS').read_text()
    refresh_checksums(tmp_path)
    assert (tmp_path/'SHA256SUMS').read_text()==first
    assert first.startswith(hashlib.sha256(b'test').hexdigest())


def test_full_cal_collector_on_synthetic_10000_rows(tmp_path):
    import csv
    from run_text_ablation import collect, INCUMBENT, MATCHED, VARIANTS
    work=tmp_path/'work'; (work/'prepared').mkdir(parents=True)
    matrix=(.75*np.eye(5)+.05*np.ones((5,5))).tolist()
    meta=work/'prepared/metadata.json';meta.write_text(json.dumps({'transition':matrix}))
    meta_hash=hashlib.sha256(meta.read_bytes()).hexdigest()
    sample=tmp_path/'sample.csv'
    with sample.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','rating']);w.writerows((i,3) for i in range(10000))
    architecture={'objective':'forward','backbone':'hashcnn','vocab_size':128,'max_tokens':160,
                  'embedding_dim':64,'cnn_channels':64,'hidden_dim':128,'dense_dim':12,
                  'dropout':.2,'kernels':[3,5],'label_mean':3.}
    args={'max_epochs':4,'batch_size':128,'weight_decay':.0001,'workers':2,'threads':4,
          'shuffle_buffer':4096,'amp':True}
    epochs=[{'epoch':i,'learning_rate':.001 if i<4 else .0005,'train_records':10,
             'calibration':{'corrected_latent_risk':{'estimated_latent_mse':1/i}}}
            for i in range(1,5)]
    def record(config,checkpoint):
        return {'seed':43,'metadata_sha256':meta_hash,'smoke':False,
                'evaluation_split_used_for_selection':False,'architecture':config,
                'epochs':epochs,'best_latent_epoch':4,'best_epoch':4,
                'best_latent_checkpoint':checkpoint,'checkpoint':checkpoint,'args':args,
                'parameter_count':parameter_count(config)['trainable_parameters']}
    configs={n:architecture for n in INCUMBENT+[MATCHED]}
    for name,layout,pool in VARIANTS:
        configs[name+'_latent']={**architecture,'text_layout':layout,'pooling':pool}
    for name,config in configs.items():
        source_name='E4_long8_seed43' if name==MATCHED else (name[:-7] if name.endswith('_latent') else name)
        source=work/'runs'/source_name;source.mkdir(parents=True,exist_ok=True)
        checkpoint=str(source/'best_latent.pt');m=record(config,checkpoint)
        (source/'metrics.json').write_text(json.dumps(m))
        folder=work/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
        (folder/'metrics.json').write_text(json.dumps(m))
        for split in ['cal','test']:
            with (folder/('predictions_'+split+'.csv')).open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['id','rating','title_star','p_observed','p_latent'])
                ids=range(9,-1,-1) if split=='cal' else range(9999,-1,-1)
                for i in ids:
                    y=1+i%5
                    latent=float(y) if name.startswith('T3') else (3. if not name.startswith('T') else 2.8)
                    observed=.75*latent+.75
                    w.writerow([i,y if split=='cal' else '',0,observed,latent])
    out=tmp_path/'out';out.mkdir()
    result=collect(SimpleNamespace(work=work,out=out,sample=sample))
    assert result['best_new_name']=='T3_separate_meanmax_seed43_latent'
    assert len(result['ablation_cells'])==4
    assert result['five_model_weights']==[.2]*5
    with (out/'recommended.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==10000 and [int(r['id']) for r in rows]==list(range(10000))
    for line in (out/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected
