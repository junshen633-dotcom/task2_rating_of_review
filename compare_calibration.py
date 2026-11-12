"""Development-set diagnostics. Never reads final eval or chooses on it."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def read(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    ids = np.array([int(r['id']) for r in rows], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate prediction ids')
    order = ids.argsort()
    return ids[order], np.array([[float(r['rating']), float(r['p_observed'])] for r in rows])[order]

def score(y,p):
    err=p-y
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(err**2))),
            'mae':float(np.mean(abs(err))), 'bias':float(err.mean()),
            'prediction_mean':float(p.mean()),'prediction_std':float(p.std()),
            'absolute_error_gt2_fraction':float(np.mean(abs(err)>2))}

def run(a):
    root=Path(a.root); result={'scope':'Calibration/development only, used for selection, not unbiased final evaluation','models':{}}
    reference=None
    for name in a.models:
        folder=root/'runs'/name
        if not (folder/'metrics.json').exists():
            continue
        m=json.loads((folder/'metrics.json').read_text())
        ids,values=read(folder/'predictions_cal.csv'); y,p=values.T
        if reference is None:
            reference=(ids,y,p,name)
        else:
            if not np.array_equal(ids,reference[0]) or not np.array_equal(y,reference[1]):
                raise ValueError('Models do not share the same calibration ids and labels')
        d=(p-y)**2-(reference[2]-y)**2
        se=float(d.std(ddof=1)/np.sqrt(len(d)))
        result['models'][name]={'best_epoch':m['best_epoch'],'overall':score(y,p),
            'per_rating':{str(k):score(y[y==k],p[y==k]) for k in range(1,6)},
            'epoch_history':m['epochs'],
            'paired_mse_difference_vs_'+reference[3]:{'mean':float(d.mean()),'approx_95pct_ci':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],
                'caveat':'IID-row approximation on selected development predictions; not a post-selection or clustered inference guarantee'},
            'test_inference':m['predictions'].get('test'),
            'hardware':m['hardware'],'offline_seconds':m['total_offline_seconds']}
    if not result['models']:
        raise ValueError('No completed models')
    output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:{'best_epoch':v['best_epoch'],**v['overall']} for k,v in result['models'].items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--models',nargs='+',default=['E1','E2','E3']);p.add_argument('--out',required=True)
    run(p.parse_args())
