"""Paired comparison on identical development holdout references; no threshold tuning."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from features import base_features,features_v2
from evaluation import entity_f05,evaluate
from pipeline import save_json


def main():
    old=json.loads(Path('artifacts/v1/training_candidates.json').read_text())
    new=json.loads(Path('artifacts/v2/training_candidates.json').read_text())
    ids=sorted(s for s in old['refs'] if int(hashlib.sha256(('model-v1:'+s).encode()).hexdigest(),16)%10>=8)
    if set(old['refs'])!=set(new['refs']):
        raise ValueError('Reference sample mismatch')
    truth={s:set(t) for s,t in old['truth'].items()}
    report={}; values=[]
    with threadpool_limits(limits=4):
        for name,data,path,feature_fn in [('v1',old,'artifacts/v1/model.pkl',base_features),('v2',new,'artifacts/v2/model.pkl',features_v2)]:
            with open(path,'rb') as f: bundle=pickle.load(f)
            pred={}
            for sid in ids:
                rows=data['candidates'][sid]; mids=list(rows)
                probs=bundle['model'].predict_proba(np.asarray([feature_fn(data['refs'][sid],rows[mid]) for mid in mids],dtype=np.float32))[:,1] if mids else []
                pred[sid]={mid for mid,p in zip(mids,probs) if p>=bundle['threshold']}
            report[name]=evaluate(ids,truth,pred,data['refs'])
            values.append(np.array([entity_f05(truth[s],pred[s]) for s in ids]))
    delta=values[1]-values[0]
    rng=np.random.default_rng(2026)
    bootstrap=np.mean(delta[rng.integers(0,len(ids),size=(2000,len(ids)))],axis=1)
    feature_report=Path('reports/feature_only/model_metrics.json')
    if feature_report.exists():
        report['feature_only']=json.loads(feature_report.read_text())['holdout']
    report['paired_macro_f05_gain']=float(delta.mean())
    report['paired_bootstrap_95_interval']=list(map(float,np.quantile(bootstrap,[.025,.975])))
    report['caveat']='Reused development holdout; interval measures sampled-reference variability, not model-selection or country-shift uncertainty. Fresh validation is required before final claims.'
    save_json(Path('reports/model_comparison.json'),report)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
