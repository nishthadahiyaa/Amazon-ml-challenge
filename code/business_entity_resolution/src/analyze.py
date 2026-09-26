"""Streaming audit and reproducible sampled baseline; Python standard library only."""
import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        yield from csv.DictReader(f, delimiter='\t')


def norm(s):
    s = unicodedata.normalize('NFKC', s).casefold().replace('&', ' and ')
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in s).split())


def keys(r):
    # Keep scripts and country labels intact; no ID-based matching features.
    name, addr, country = norm(r['business_name']), norm(r['business_address']), r['country']
    result = []
    if name:
        result.append((country, 'name', name))
    if addr:
        result.append((country, 'address', addr))
    return result


def score(a, b):
    n1, n2 = norm(a['business_name']), norm(b['business_name'])
    a1, a2 = norm(a['business_address']), norm(b['business_address'])
    n = SequenceMatcher(None, n1, n2, autojunk=False).ratio() if n1 and n2 else 0
    ad = SequenceMatcher(None, a1, a2, autojunk=False).ratio() if a1 and a2 else 0
    return 0.55*n + 0.45*ad


def entity_f(truth, pred):
    if not truth:
        return float(not pred)
    return 1.25*len(truth & pred)/(0.25*len(truth)+len(pred))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('student_resource/dataset'))
    p.add_argument('--reports', type=Path, default=Path('reports'))
    p.add_argument('--sample-size', type=int, default=4000)
    args = p.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2026)
    profiles, sample = {}, []
    # Uniform reservoir over the full reference file, not a potentially sorted prefix.
    for path in sorted(args.data.glob('*/*source*.tsv')):
        countries, missing, non_ascii = Counter(), Counter(), Counter()
        lengths = Counter()
        n = 0
        for n, r in enumerate(rows(path), 1):
            if None in r or any(v is None for v in r.values()):
                raise ValueError(f'Malformed row {n}: {path}')
            countries[r['country']] += 1
            for col in ('business_name', 'business_address', 'country'):
                missing[col] += int(not r[col].strip())
                non_ascii[col] += int(not r[col].isascii())
                lengths[col] += len(r[col])
            if path.name == 'train_source1.tsv':
                if len(sample) < args.sample_size:
                    sample.append(r)
                else:
                    j = rng.randrange(n)
                    if j < args.sample_size:
                        sample[j] = r
        profiles[path.name] = dict(rows=n, countries=dict(countries), missing=dict(missing),
                                  non_ascii=dict(non_ascii), mean_length={k:v/n for k,v in lengths.items()})
        print(path.name, n, dict(countries), flush=True)
    refs = {r['entity_id']:r for r in sample}
    truth, histogram, total, duplicate_targets = {}, Counter(), 0, 0
    for r in rows(args.data/'train/train_ground_truth.tsv'):
        ids = r['matched_entity_ids'].split(',') if r['matched_entity_ids'] else []
        histogram[len(ids)] += 1
        duplicate_targets += len(ids)-len(set(ids))
        total += 1
        if r['source1_entity_id'] in refs:
            if r['source1_entity_id'] in truth:
                raise ValueError('Duplicate sampled ground-truth reference')
            truth[r['source1_entity_id']] = set(ids)
    assert set(refs) == set(truth), 'Missing sampled ground truth'
    profiles['ground_truth'] = dict(rows=total, match_count_histogram=dict(histogram), duplicate_targets=duplicate_targets)
    (args.reports/'dataset_profile.json').write_text(json.dumps(profiles, indent=2))
    index = defaultdict(set)
    for sid, r in refs.items():
        for key in keys(r):
            index[key].add(sid)
    candidates = defaultdict(dict)
    for source in (2,3):
        for r in rows(args.data/f'train/train_source{source}.tsv'):
            sids = set()
            for key in keys(r):
                sids.update(index.get(key, ()))
            for sid in sids:
                candidates[sid][r['entity_id']] = score(refs[sid], r)
        print(f'Candidate scan source {source} complete', flush=True)
    tune = [sid for sid in refs if int(hashlib.sha256(sid.encode()).hexdigest(),16)%2 == 0]
    held = sorted(set(refs)-set(tune))
    def evaluate(ids, threshold):
        values, tp, fp, fn, single_correct, singles = [], 0, 0, 0, 0, 0
        by_country = defaultdict(list)
        for sid in ids:
            pred = {mid for mid,s in candidates[sid].items() if s >= threshold}
            t = truth[sid]
            f = entity_f(t,pred)
            values.append(f); by_country[refs[sid]['country']].append(f)
            tp += len(t & pred); fp += len(pred-t); fn += len(t-pred)
            singles += int(not t); single_correct += int(not t and not pred)
        return dict(entities=len(ids), macro_f05=sum(values)/len(values),
                    precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0,
                    singleton_accuracy=single_correct/singles if singles else None,
                    by_country={c:sum(v)/len(v) for c,v in by_country.items()})
    sweep = {str(t/100):evaluate(tune,t/100)['macro_f05'] for t in range(55,101)}
    threshold = float(max(sweep,key=lambda x:(sweep[x],float(x))))
    hits = sum(len(truth[s] & candidates[s].keys()) for s in refs)
    positives = sum(map(len,truth.values()))
    result = dict(seed=2026, sample_entities=len(refs), blocking='country + exact normalized name OR address',
                  candidates=sum(map(len,candidates.values())), blocking_recall=hits/positives,
                  threshold=threshold, tuning=evaluate(tune,threshold), holdout=evaluate(held,threshold),
                  empty_baseline_holdout=sum(not truth[s] for s in held)/len(held), threshold_sweep=sweep,
                  limitations=['Exact blocking misses fuzzy-only matches.', 'France has no labeled validation data.',
                               'Baseline is a rule score, not a trained ML model.', 'Audit does not yet verify global ID uniqueness or all label references.'])
    (args.reports/'baseline_metrics.json').write_text(json.dumps(result,indent=2))
    with (args.reports/'baseline_errors.tsv').open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t'); w.writerow(['source1_entity_id','country','business_name','business_address','true_ids','predicted_ids','missed_by_blocking'])
        for sid in held:
            pred={mid for mid,s in candidates[sid].items() if s>=threshold}
            if pred!=truth[sid]:
                r=refs[sid]; w.writerow([sid,r['country'],r['business_name'],r['business_address'],','.join(sorted(truth[sid])),','.join(sorted(pred)),','.join(sorted(truth[sid]-candidates[sid].keys()))])
    print(json.dumps({k:v for k,v in result.items() if k!='threshold_sweep'},indent=2))

if __name__ == '__main__':
    main()
