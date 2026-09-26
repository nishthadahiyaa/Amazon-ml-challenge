"""Streaming vendor search against an in-memory reference index, bounded top-k."""
import csv
import heapq
from collections import defaultdict
from preprocessing import prepare, blocking_keys, base_blocking_keys
from features import retrieval_score


def read_rows(path):
    with open(path,encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed TSV row in {path}: {reader.line_num}')
            yield row


def retrieve(refs, vendor_paths, top_k=80, max_block=200, blocking_version=2):
    if blocking_version not in (1,2):
        raise ValueError("Unsupported blocking version")
    key_fn=base_blocking_keys if blocking_version==1 else blocking_keys
    index=defaultdict(list)
    for sid,r in refs.items():
        for key in key_fn(r):
            index[key].append(sid)
    # Ignore broad blocks rather than silently taking the first references.
    dropped=sum(len(v)>max_block for v in index.values())
    index={k:v for k,v in index.items() if len(v)<=max_block}
    print(f'Indexed {len(refs):,} references, {len(index):,} keys; dropped {dropped:,} broad keys',flush=True)
    heaps={sid:[] for sid in refs}
    sequence=0
    for path in vendor_paths:
        for i,row in enumerate(read_rows(path),1):
            r=prepare(row)
            hits=set()
            for key in key_fn(r):
                hits.update(index.get(key,()))
            for sid in hits:
                value=retrieval_score(refs[sid],r)
                sequence+=1
                entry=(value,r['id'],sequence,r)
                h=heaps[sid]
                if len(h)<top_k:
                    heapq.heappush(h,entry)
                elif entry[:2]>h[0][:2]:
                    heapq.heapreplace(h,entry)
            if i%1000000==0:
                print(f'{path.name}: scanned {i:,}',flush=True)
    # Duplicate vendor IDs are input errors, not independent candidates.
    result={}
    for sid,h in heaps.items():
        result[sid]={r['id']:r for _,_,_,r in sorted(h,reverse=True)}
    return result
