"""Pair comparisons; IDs and country labels are never categorical model features."""
from rapidfuzz import fuzz
import unicodedata
from functools import lru_cache
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES = ['name_ratio','name_token_sort','name_token_set','name_jaro',
                 'core_ratio','address_ratio','address_token_sort','address_token_set',
                 'name_jaccard','address_jaccard','number_jaccard','number_conflict',
                 'name_exact','address_exact','address_missing','name_length_ratio',
                 'address_length_ratio','same_country']


def jaccard(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if a or b else 0.0


def length_ratio(a,b):
    return min(len(a),len(b))/max(len(a),len(b)) if a and b else 0.0


def base_features(a,b):
    n,m=a['name'],b['name']; x,y=a['address'],b['address']
    return [fuzz.ratio(n,m)/100, fuzz.token_sort_ratio(n,m)/100,
            fuzz.token_set_ratio(n,m)/100, JaroWinkler.normalized_similarity(n,m),
            fuzz.ratio(a['core'],b['core'])/100,
            fuzz.ratio(x,y)/100 if x and y else 0,
            fuzz.token_sort_ratio(x,y)/100 if x and y else 0,
            fuzz.token_set_ratio(x,y)/100 if x and y else 0,
            jaccard(a['tokens'],b['tokens']),jaccard(x.split(),y.split()),
            jaccard(a['numbers'],b['numbers']),
            float(bool(a['numbers'] and b['numbers']) and not set(a['numbers'])&set(b['numbers'])),
            float(bool(n) and n==m),float(bool(x) and x==y),float(not x or not y),
            length_ratio(n,m),length_ratio(x,y),float(a['country']==b['country'])]


def retrieval_score(a,b):
    return (0.55*fuzz.token_sort_ratio(a['core'],b['core'])+
            0.45*fuzz.token_sort_ratio(a['address'],b['address']))/100


FEATURE_NAMES += ['name_fuzzy_coverage_ab','name_fuzzy_coverage_ba',
                  'address_fuzzy_coverage_ab','address_fuzzy_coverage_ba',
                  'folded_name_ratio','folded_address_ratio','name_phonetic_jaccard',
                  'number_overlap_count','name_overlap_count','address_overlap_count',
                  'address_numbers_missing_a','address_numbers_missing_b',
                  'core_token_sort','name_partial','address_partial',
                  'name_nonlatin_a','name_nonlatin_b','name_script_disjoint']


@lru_cache(maxsize=100000)
def scripts(text):
    return frozenset(unicodedata.name(c, "UNKNOWN").split()[0] for c in text if c.isalpha())


def fuzzy_coverage(a,b):
    if not a or not b: return 0.0
    return sum(max(fuzz.ratio(x,y) for y in b) for x in a)/(100*len(a))


def features(a,b):
    from preprocessing import fold_latin,soundex
    an,bn=a['tokens'],b['tokens']; ax,bx=a['address'].split(),b['address'].split()
    # Bound fuzzy-token comparisons for unusually long records.
    an,bn=an[:20],bn[:20]; ax,bx=ax[:30],bx[:30]
    pa={soundex(fold_latin(t)) for t in an}-{''}; pb={soundex(fold_latin(t)) for t in bn}-{''}
    return base_features(a,b)+[
        fuzzy_coverage(an,bn),fuzzy_coverage(bn,an),fuzzy_coverage(ax,bx),fuzzy_coverage(bx,ax),
        fuzz.ratio(fold_latin(a['name']),fold_latin(b['name']))/100,
        fuzz.ratio(fold_latin(a['address']),fold_latin(b['address']))/100 if ax and bx else 0,
        jaccard(pa,pb),len(set(a['numbers'])&set(b['numbers'])),len(set(an)&set(bn)),len(set(ax)&set(bx)),
        float(not a['numbers']),float(not b['numbers']),fuzz.token_sort_ratio(a['core'],b['core'])/100,
        fuzz.partial_ratio(a['name'],b['name'])/100 if an and bn else 0,
        fuzz.partial_ratio(a['address'],b['address'])/100 if ax and bx else 0,
        float(bool(scripts(a['name'])-{'LATIN'})),float(bool(scripts(b['name'])-{'LATIN'})),
        float(bool(scripts(a['name']) and scripts(b['name'])) and not scripts(a['name'])&scripts(b['name']))]


# V3 keeps the previous comparison vector available for old model artifacts.
FEATURE_NAMES_V2 = FEATURE_NAMES.copy()
features_v2 = features
FEATURE_NAMES += ['canonical_number_jaccard', 'canonical_number_conflict',
                  'compact_name_ratio']


def canonical_numbers(record):
    return {str(int(n)) for n in record['numbers'] if len(n) <= 9}


def features(a,b):
    an,bn=canonical_numbers(a),canonical_numbers(b)
    return features_v2(a,b)+[
        jaccard(an,bn),float(bool(an and bn) and not an & bn),
        fuzz.ratio(''.join(a['core'].split()), ''.join(b['core'].split()))/100]
