"""Compare candidate sets on the same reference sample and grouped holdout."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from evaluation import entity_f05


def stats(data, ids):
    by=defaultdict(list)
    for sid in ids:
        truth=set(data['truth'][sid]); cands=set(data['candidates'][sid])
        by['all'].append((truth,cands))
        by[data['refs'][sid]['country']].append((truth,cands))
    result={}
    for group, pairs in by.items():
        positives=sum(len(t) for t,_ in pairs)
        result[group]={'references':len(pairs),
                       'candidate_pairs':sum(len(c) for _,c in pairs),
                       'true_links':positives,
                       'retained_true_links':sum(len(t&c) for t,c in pairs),
                       'link_recall':sum(len(t&c) for t,c in pairs)/positives if positives else None,
                       'oracle_macro_f05':sum(entity_f05(t,t&c) for t,c in pairs)/len(pairs)}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,default=Path('artifacts/v2/training_candidates.json'))
    p.add_argument('--new',type=Path,default=Path('artifacts/v3/training_candidates.json'))
    p.add_argument('--output',type=Path,default=Path('reports/candidate_comparison_v3.json'))
    args=p.parse_args()
    old=json.loads(args.baseline.read_text()); new=json.loads(args.new.read_text())
    if set(old['refs'])!=set(new['refs']) or old['truth']!=new['truth']:
        raise ValueError('Candidate sets do not share the same references and labels')
    holdout=sorted(s for s in old['refs'] if int(hashlib.sha256(('model-v1:'+s).encode()).hexdigest(),16)%10>=8)
    result={'baseline':stats(old,holdout),'new':stats(new,holdout)}
    result['lost_old_candidates']=sum(len(set(old['candidates'][s])-new['candidates'][s].keys()) for s in holdout)
    result['new_true_links']=sum(len((set(new['candidates'][s])-old['candidates'][s].keys())&set(new['truth'][s])) for s in holdout)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
