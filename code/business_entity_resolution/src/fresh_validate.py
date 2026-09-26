"""A new, disjoint reference sample to check an already-selected model pair."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from pipeline import sample_refs,save_json
from blocking import read_rows
from blocking_v3 import retrieve as retrieve_v3
from features import FEATURE_NAMES,features,features_v2,base_features
from evaluation import evaluate,entity_f05


def score(bundle,refs,candidates,truth,threads):
    if bundle['features']==FEATURE_NAMES: feature_fn=features
    elif bundle['features']==FEATURE_NAMES[:36]: feature_fn=features_v2
    elif bundle['features']==FEATURE_NAMES[:18]: feature_fn=base_features
    else: raise ValueError('Unknown model feature schema')
    predictions={}
    with threadpool_limits(limits=threads):
        for sid,ref in refs.items():
            targets=candidates[sid]; mids=list(targets)
            if mids:
                probs=bundle['model'].predict_proba(np.asarray(
                    [feature_fn(ref,targets[mid]) for mid in mids],dtype=np.float32))[:,1]
            else:
                probs=[]
            threshold=bundle.get('country_thresholds',{}).get(ref['country'],bundle['threshold'])
            predictions[sid]={mid for mid,p in zip(mids,probs) if p>=threshold}
    result=evaluate(list(refs),truth,predictions,refs)
    result['candidate_pairs']=sum(len(c) for c in candidates.values())
    result['blocking_recall']=sum(len(truth[s]&candidates[s].keys()) for s in refs)/sum(map(len,truth.values()))
    result['candidate_oracle_macro_f05']=sum(entity_f05(truth[s],truth[s]&candidates[s].keys()) for s in refs)/len(refs)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('student_resource/dataset'))
    p.add_argument('--exclude-cache',type=Path,default=Path('artifacts/v2/training_candidates.json'))
    p.add_argument('--baseline-model',type=Path,default=Path('artifacts/v2/model.pkl'))
    p.add_argument('--new-model',type=Path,default=Path('artifacts/v3/model.pkl'))
    p.add_argument('--output',type=Path,default=Path('reports/fresh_validation_v3.json'))
    p.add_argument('--sample-size',type=int,default=2000)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    old_ids=set(json.loads(args.exclude_cache.read_text())['refs'])
    pool=sample_refs(args.data/'train/train_source1.tsv',args.sample_size+50,5719)
    refs={sid:r for sid,r in pool.items() if sid not in old_ids}
    refs=dict(list(refs.items())[:args.sample_size])
    if len(refs)<args.sample_size: raise ValueError('Not enough disjoint references')
    truth={}
    for row in read_rows(args.data/'train/train_ground_truth.tsv'):
        sid=row['source1_entity_id']
        if sid in refs: truth[sid]=set(filter(None,row['matched_entity_ids'].split(',')))
    if len(truth)!=len(refs): raise ValueError('Missing labels for fresh references')
    with args.baseline_model.open('rb') as f: old=pickle.load(f)
    with args.new_model.open('rb') as f: new=pickle.load(f)
    if new['config']['version']!=3: raise ValueError('Expected version 3 retrieval for new model')
    candidates,baseline=retrieve_v3(refs,[args.data/f'train/train_source{s}.tsv' for s in (2,3)],
                                     top_k=new['config']['top_k'],max_block=new['config']['max_block'],include_baseline=True)
    # Baseline channel is exactly the previous retrieval policy with these batch limits.
    report={'sample_seed':5719,'excluded_previous_references':len(old_ids),
            'baseline':score(old,refs,baseline,truth,args.threads),
            'new':score(new,refs,candidates,truth,args.threads)}
    report['macro_f05_gain']=report['new']['macro_f05']-report['baseline']['macro_f05']
    save_json(args.output,report)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
