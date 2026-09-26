#!/usr/bin/env python3
"""CLI: prepare training candidates, train/evaluate, predict, package."""
import argparse
import csv
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from blocking import read_rows, retrieve
from preprocessing import prepare
from features import FEATURE_NAMES, features, features_v2, base_features
from evaluation import evaluate


def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    temp.replace(path)


def sample_refs(path,n,seed):
    rng=random.Random(seed); sample=[]
    for i,row in enumerate(read_rows(path),1):
        if len(sample)<n:
            sample.append(row)
        else:
            j=rng.randrange(i)
            if j<n:
                sample[j]=row
    return {r['entity_id']:prepare(r) for r in sample}


def prepare_training(args):
    if args.reuse_reference_cache:
        saved=json.loads(args.reuse_reference_cache.read_text())
        refs=saved['refs']
        truth={sid:set(ids) for sid,ids in saved['truth'].items()}
    else:
        refs=sample_refs(args.data/'train/train_source1.tsv',args.sample_size,args.seed)
        truth={}
    if not args.reuse_reference_cache:
        for row in read_rows(args.data/'train/train_ground_truth.tsv'):
            sid=row['source1_entity_id']
            if sid in refs:
                if sid in truth:
                    raise ValueError('Duplicate ground truth reference')
                truth[sid]=set(filter(None,row['matched_entity_ids'].split(',')))
    if set(refs)!=set(truth):
        raise ValueError('Missing ground truth references')
    if args.blocking_version == 3:
        from blocking_v3 import retrieve as retrieve_v3
        candidates=retrieve_v3(refs,[args.data/f'train/train_source{s}.tsv' for s in (2,3)],args.top_k,args.max_block)
    else:
        candidates=retrieve(refs,[args.data/f'train/train_source{s}.tsv' for s in (2,3)],args.top_k,args.max_block,blocking_version=args.blocking_version)
    save_json(args.artifacts/'training_candidates.json',{
        'refs':refs,'truth':{s:sorted(t) for s,t in truth.items()},'candidates':candidates,
        'config':{'seed':args.seed,'top_k':args.top_k,'max_block':args.max_block,
                  'sample_size':args.sample_size,'version':args.blocking_version}})


def train(args):
    d=json.loads((args.artifacts/'training_candidates.json').read_text())
    refs,candidates=d['refs'],d['candidates']; truth={s:set(t) for s,t in d['truth'].items()}
    groups={'train':[],'tune':[],'holdout':[]}
    for sid in refs:
        bucket=int(hashlib.sha256(('model-v1:'+sid).encode()).hexdigest(),16)%10
        groups['train' if bucket<6 else 'tune' if bucket<8 else 'holdout'].append(sid)
    if any(not x for x in groups.values()):
        raise ValueError('Sample too small for three reference splits')
    # Fail rather than silently leak a vendor shared by labeled reference entities.
    owners={}
    for group,ids in groups.items():
        for sid in ids:
            for mid in truth[sid]:
                if mid in owners and owners[mid]!=group:
                    raise ValueError('Label component spans splits; implement component-group splitting')
                owners[mid]=group
    x=[]; y=[]
    for sid in groups['train']:
        for mid,r in candidates[sid].items():
            x.append(features(refs[sid],r)); y.append(int(mid in truth[sid]))
    if len(set(y))<2:
        raise ValueError('Training candidates must contain both classes')
    model_kind=getattr(args,'model','histgb')
    if model_kind=='lightgbm':
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(n_estimators=420,num_leaves=31,learning_rate=0.04,
                             min_child_samples=35,reg_lambda=3.0,
                             subsample=0.85,subsample_freq=1,colsample_bytree=0.9,
                             random_state=2026,n_jobs=args.threads,verbosity=-1)
    else:
        model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=0.08,
                                            l2_regularization=3,min_samples_leaf=30,
                                            early_stopping=False,random_state=2026)
    with threadpool_limits(limits=args.threads):
        model.fit(np.asarray(x,dtype=np.float32),np.asarray(y))
        scores={}
        for group in ('tune','holdout'):
            for sid in groups[group]:
                mids=list(candidates[sid])
                prob=model.predict_proba(np.asarray([features(refs[sid],candidates[sid][mid]) for mid in mids],dtype=np.float32))[:,1] if mids else []
                scores[sid]=dict(zip(mids,map(float,prob)))
    country_thresholds={}
    def predictions(ids,t):
        return {sid:{mid for mid,s in scores[sid].items()
                     if s>=country_thresholds.get(refs[sid]['country'],t)} for sid in ids}
    sweep=[]
    for t in np.linspace(0.05,0.99,95):
        metrics=evaluate(groups['tune'],truth,predictions(groups['tune'],t),refs)
        sweep.append((float(t),metrics['macro_f05']))
    threshold=max(sweep,key=lambda v:(v[1],v[0]))[0]
    if getattr(args,'threshold_mode','global')=='country':
        for country in sorted({refs[s]['country'] for s in groups['tune']}):
            subset=[s for s in groups['tune'] if refs[s]['country']==country]
            if len(subset)>=100:
                local=[(t,evaluate(subset,truth,predictions(subset,t),refs)['macro_f05'])
                       for t in np.linspace(0.05,0.99,95)]
                country_thresholds[country]=float(max(local,key=lambda v:(v[1],v[0]))[0])
    report={'model':type(model).__name__,'features':FEATURE_NAMES,
            'config':d['config'],'threshold':threshold,'country_thresholds':country_thresholds,'training_pairs':len(y),'training_positives':sum(y),
            'split_sizes':{k:len(v) for k,v in groups.items()},'threshold_sweep':sweep}
    for group in ('tune','holdout'):
        ids=groups[group]
        report[group]=evaluate(ids,truth,predictions(ids,threshold),refs)
        positives=sum(len(truth[s]) for s in ids)
        report[group]['blocking_recall']=sum(len(truth[s]&candidates[s].keys()) for s in ids)/positives if positives else None
        report[group]['candidate_pairs']=sum(len(candidates[s]) for s in ids)
    args.artifacts.mkdir(parents=True,exist_ok=True)
    with (args.artifacts/'model.pkl').open('wb') as f:
        pickle.dump({'model':model,'threshold':threshold,'country_thresholds':country_thresholds,
                     'config':d['config'],'features':FEATURE_NAMES},f)
    save_json(args.reports/'model_metrics.json',report)
    with (args.reports/'model_errors.tsv').open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t'); w.writerow(['source1_entity_id','country','false_positive_ids','false_negative_ids','missed_by_blocking'])
        pred=predictions(groups['holdout'],threshold)
        for sid in groups['holdout']:
            if pred[sid]!=truth[sid]:
                w.writerow([sid,refs[sid]['country'],','.join(sorted(pred[sid]-truth[sid])),','.join(sorted(truth[sid]-pred[sid])),','.join(sorted(truth[sid]-candidates[sid].keys()))])
    print(json.dumps({k:v for k,v in report.items() if k not in ('threshold_sweep','features')},indent=2),flush=True)


def predict(args):
    # Only load model artifacts produced locally by this project.
    with (args.artifacts/'model.pkl').open('rb') as f:
        bundle=pickle.load(f)
    if bundle['features']==FEATURE_NAMES:
        inference_features=features
    elif bundle['features']==FEATURE_NAMES[:36]:
        inference_features=features_v2
    elif bundle['features']==FEATURE_NAMES[:18]:
        inference_features=base_features
    else:
        raise ValueError('Model feature schema mismatch')
    args.output.mkdir(parents=True,exist_ok=True)
    paths=[args.output/'matching_results.tsv',args.output/'candidate_pairs.tsv']
    pending=[p.with_suffix('.tsv.tmp') for p in paths]
    source=iter(read_rows(args.data/'test/test_source1.tsv'))
    config=bundle['config']; count=0
    with pending[0].open('w',newline='') as mf,pending[1].open('w',newline='') as cf,threadpool_limits(limits=args.threads):
        mw,cw=csv.writer(mf,delimiter='\t'),csv.writer(cf,delimiter='\t')
        mw.writerow(['source1_entity_id','matched_entity_ids']); cw.writerow(['source1_entity_id','candidate_entity_ids'])
        while True:
            refs={}
            for _ in range(args.reference_batch_size):
                row=next(source,None)
                if row is None:
                    break
                r=prepare(row)
                if r['id'] in refs:
                    raise ValueError('Duplicate reference ID in batch')
                refs[r['id']]=r
            if not refs:
                break
            if config.get('version',1)==3:
                from blocking_v3 import retrieve as retrieve_v3
                candidates=retrieve_v3(refs,[args.data/f'test/test_source{s}.tsv' for s in (2,3)],config['top_k'],config['max_block'])
            else:
                candidates=retrieve(refs,[args.data/f'test/test_source{s}.tsv' for s in (2,3)],config['top_k'],config['max_block'],blocking_version=config.get('version',1))
            for sid in refs:
                mids=sorted(candidates[sid])
                # Export exactly the final candidate set that receives inference.
                cw.writerow([sid,','.join(mids)])
                probs=bundle['model'].predict_proba(np.asarray([inference_features(refs[sid],candidates[sid][mid]) for mid in mids],dtype=np.float32))[:,1] if mids else []
                limit=bundle.get('country_thresholds',{}).get(refs[sid]['country'],bundle['threshold'])
                matches=[mid for mid,p in zip(mids,probs) if p>=limit]
                mw.writerow([sid,','.join(matches)])
            count+=len(refs); print(f'Wrote {count:,} reference entities',flush=True)
    for tmp,path in zip(pending,paths):
        tmp.replace(path)
    save_json(args.output/'run_metadata.json',{'reference_rows':count,'config':config,'threshold':bundle['threshold'],
        'reference_batch_size':args.reference_batch_size,
        'note':'Block frequency cap is applied within each reference batch; batch size can change retrieval.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','train','predict'])
    p.add_argument('--data',type=Path,default=Path('student_resource/dataset'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    p.add_argument('--reports',type=Path,default=Path('reports'))
    p.add_argument('--output',type=Path,default=Path('output'))
    p.add_argument('--sample-size',type=int,default=4000)
    p.add_argument('--blocking-version',type=int,choices=[1,2,3],default=3)
    p.add_argument('--model',choices=['histgb','lightgbm'],default='histgb')
    p.add_argument('--threshold-mode',choices=['global','country'],default='global')
    p.add_argument('--reuse-reference-cache',type=Path,default=None)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--top-k',type=int,default=120)
    p.add_argument('--max-block',type=int,default=200)
    p.add_argument('--reference-batch-size',type=int,default=10000)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    for field in ('sample_size','top_k','max_block','reference_batch_size','threads'):
        if getattr(args,field)<1:
            p.error(f'--{field.replace("_","-")} must be positive')
    {'prepare':prepare_training,'train':train,'predict':predict}[args.command](args)

if __name__=='__main__':
    main()
