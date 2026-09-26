"""Competition macro F0.5; never class-macro F-beta."""
from collections import defaultdict


def entity_f05(truth,pred):
    if not truth:
        return float(not pred)
    return 1.25*len(truth&pred)/(0.25*len(truth)+len(pred))


def evaluate(ids,truth,predictions,refs):
    tp=fp=fn=singletons=correct=0
    scores=[]; countries=defaultdict(list)
    for sid in ids:
        t,p=truth[sid],predictions.get(sid,set())
        s=entity_f05(t,p); scores.append(s); countries[refs[sid]['country']].append(s)
        tp+=len(t&p); fp+=len(p-t); fn+=len(t-p)
        singletons+=int(not t); correct+=int(not t and not p)
    return {'entities':len(ids),'macro_f05':sum(scores)/len(scores),
            'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,
            'singleton_accuracy':correct/singletons if singletons else None,
            'by_country':{k:sum(v)/len(v) for k,v in countries.items()}}
