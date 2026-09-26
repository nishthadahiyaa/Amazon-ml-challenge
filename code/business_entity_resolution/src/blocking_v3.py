"""Multi-channel streaming retrieval with protected name/address/gram slots.

The baseline top-k is retained as one channel, so the larger candidate set is a
superset of v2 when reference batches and block caps are identical. Only the
provided source text is indexed; no external identity data is used.
"""
import heapq
from collections import Counter, defaultdict

from preprocessing import blocking_keys, prepare
from rapidfuzz import fuzz


def canonical_numbers(record):
    return {str(int(n)) for n in record['numbers'] if len(n) <= 9}


def grams(value):
    value = ''.join(value.split())
    return {value[i:i + 4] for i in range(max(0, len(value) - 3))}


def bounded_index(refs, token_fn, cap):
    index = defaultdict(list)
    for sid, record in refs.items():
        for token in token_fn(record):
            index[(record['country'], token)].append(sid)
    return {key: ids for key, ids in index.items() if len(ids) <= cap}


def push(heap, record, value, limit, sequence):
    item = (value, record['id'], sequence, record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def retrieve(refs, vendor_paths, top_k=120, max_block=200, include_baseline=False):
    from blocking import read_rows

    base_index = defaultdict(list)
    for sid, record in refs.items():
        for key in blocking_keys(record):
            base_index[key].append(sid)
    base_index = {key: ids for key, ids in base_index.items() if len(ids) <= max_block}

    def address_tokens(r):
        return {t for t in r['address'].split() if len(t) >= 4 and t.isalpha()}

    def name_tokens(r):
        return {t for t in r['tokens'] if len(t) >= 4 and t.isalpha()}

    address_index = bounded_index(refs, address_tokens, 40)
    name_index = bounded_index(refs, name_tokens, 30)
    address_gram_index = bounded_index(refs, lambda r: grams(r['address']), 25)
    name_gram_index = bounded_index(refs, lambda r: grams(r['core']), 25)
    # Reserve slots for independent evidence. Do not let common names consume all
    # candidates before address-only or cross-script candidates are considered.
    limits = {'base': top_k, 'address': 60, 'name': 30, 'gram': 30}
    heaps = {sid: {channel: [] for channel in limits} for sid in refs}
    sequence = 0
    print(f'V3 indexed {len(refs):,} references: {len(base_index):,} base, '
          f'{len(address_index):,} address tokens, {len(name_index):,} name tokens, '
          f'{len(address_gram_index)+len(name_gram_index):,} grams', flush=True)

    for path in vendor_paths:
        for i, row in enumerate(read_rows(path), 1):
            record = prepare(row)
            country = record['country']
            base_hits = set()
            for key in blocking_keys(record):
                base_hits.update(base_index.get(key, ()))

            address_hits = Counter()
            for token in address_tokens(record):
                for sid in address_index.get((country, token), ()):
                    address_hits[sid] += 1
            number_hits = canonical_numbers(record)
            address_hits = {sid for sid, shared in address_hits.items()
                            if shared >= 2 or (shared and number_hits & canonical_numbers(refs[sid]))}

            name_hits = Counter()
            for token in name_tokens(record):
                posting = name_index.get((country, token), ())
                for sid in posting:
                    name_hits[sid] += 1
            name_hits = {sid for sid, shared in name_hits.items() if shared >= 2 or
                         (shared and len(refs[sid]['tokens']) <= 2)}

            gram_hits = Counter()
            for index, value in ((name_gram_index, record['core']),
                                 (address_gram_index, record['address'])):
                postings = [index[(country, gram)] for gram in grams(value)
                            if (country, gram) in index]
                postings.sort(key=len)
                for posting in postings[:16]:
                    for sid in posting:
                        gram_hits[sid] += 1
            gram_hits = {sid for sid, shared in gram_hits.items() if shared >= 3}

            all_hits = base_hits | address_hits | name_hits | gram_hits
            for sid in all_hits:
                reference = refs[sid]
                # One pair is scored at most once per route. Feature generation
                # happens only after the final candidate union is fixed.
                n = fuzz.token_sort_ratio(reference['core'], record['core']) / 100
                a = fuzz.token_sort_ratio(reference['address'], record['address']) / 100
                sequence += 1
                if sid in base_hits:
                    push(heaps[sid]['base'], record, .55*n + .45*a, top_k, sequence)
                if sid in address_hits:
                    push(heaps[sid]['address'], record, .15*n + .85*a, 60, sequence)
                if sid in name_hits:
                    push(heaps[sid]['name'], record, .80*n + .20*a, 30, sequence)
                if sid in gram_hits:
                    push(heaps[sid]['gram'], record, .50*n + .50*a, 30, sequence)
            if i % 1000000 == 0:
                print(f'V3 {path.name}: scanned {i:,}', flush=True)

    union={sid: {record['id']: record for channel in limits
                 for _, _, _, record in heaps[sid][channel]} for sid in refs}
    if include_baseline:
        base={sid: {record['id']: record for _, _, _, record in heaps[sid]['base']}
              for sid in refs}
        return union, base
    return union
