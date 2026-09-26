"""Unicode-safe normalization and inexpensive retrieval signatures."""
import re
import unicodedata
from itertools import combinations
from functools import lru_cache

ALIASES = {'incorporated':'inc', 'corporation':'corp', 'limited':'ltd',
           'private':'pvt', 'company':'co', 'street':'st', 'road':'rd',
           'avenue':'ave', 'boulevard':'blvd', 'suite':'ste', 'apartment':'apt'}
LEGAL = {'inc','corp','ltd','pvt','co','llc','llp','plc','sarl','sas','sa','and','the'}


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold().replace('&', ' and ')
    text = ''.join(c if c.isalnum() or unicodedata.category(c).startswith('M') else ' ' for c in text)
    return ' '.join(ALIASES.get(t,t) for t in text.split())


def prepare(row):
    name = normalize(row['business_name'])
    address = normalize(row['business_address'])
    tokens = tuple(sorted(set(name.split())-LEGAL))
    return {'id':row['entity_id'], 'country':row['country'], 'name':name,
            'address':address, 'tokens':tokens, 'core':' '.join(tokens),
            'numbers':tuple(sorted(set(re.findall(r'\d+',address))))}


def base_blocking_keys(r):
    country = r['country']
    if r['core']:
        yield (country,'name',r['core'])
    if r['address']:
        yield (country,'address',' '.join(sorted(r['address'].split())))
    # Bounded per-record signatures; caps affect recall and are measured on labels.
    tokens = sorted(r['tokens'],key=lambda x:(-len(x),x))[:6]
    for a,b in combinations(sorted(tokens),2):
        yield (country,'pair',a,b)
    for token in tokens[:4]:
        for number in r['numbers'][:3]:
            yield (country,'number',token,number)


@lru_cache(maxsize=100000)
def fold_latin(text):
    """Remove Latin accents without removing combining marks from Indic scripts."""
    result=[]; latin=False
    for c in unicodedata.normalize('NFKD',text):
        if not unicodedata.category(c).startswith('M'):
            latin='LATIN' in unicodedata.name(c,'')
        if not (latin and unicodedata.category(c).startswith('M')):
            result.append(c)
    return ''.join(result)


@lru_cache(maxsize=100000)
def soundex(token):
    if not token or not token.isascii() or not token.isalpha():
        return ''
    groups={'b':'1','f':'1','p':'1','v':'1','c':'2','g':'2','j':'2','k':'2','q':'2','s':'2','x':'2','z':'2',
            'd':'3','t':'3','l':'4','m':'5','n':'5','r':'6'}
    out=token[0]; last=groups.get(token[0],'')
    for c in token[1:]:
        code=groups.get(c,'')
        if code and code!=last: out+=code
        if c not in 'hw': last=code
    return (out+'000')[:4]



def blocking_keys(r):
    yield from base_blocking_keys(r)
    country=r['country']
    tokens=sorted(r['tokens'],key=lambda x:(-len(x),x))[:4]
    numbers=r['numbers'][:3]
    phonetics=sorted({soundex(fold_latin(t)) for t in tokens}-{''})
    for code in phonetics:
        for number in numbers:
            yield (country,'phonetic_number',code,number)
    # Address-only paths recover cross-script business names and heavily corrupted names.
    address_tokens=sorted({t for t in r['address'].split() if len(t)>=5 and t.isalpha()},key=lambda x:(-len(x),x))[:4]
    for token in address_tokens:
        for number in numbers:
            yield (country,'address_number',token,number)
    for a,b in combinations(sorted(address_tokens),2):
        yield (country,'address_pair',a,b)
