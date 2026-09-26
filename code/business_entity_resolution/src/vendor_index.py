"""Experimental persistent vendor search index, built only from supplied TSV files.

The index avoids rescanning millions of vendor rows for every reference batch.
It is not yet the default candidate source: first compare its candidate oracle with
streaming retrieval on labeled data before using it for a final submission.
"""
import argparse
import sqlite3
from pathlib import Path
from preprocessing import normalize
from blocking import read_rows


def build(paths,output,batch_size=50000):
    output.parent.mkdir(parents=True,exist_ok=True)
    pending=output.with_suffix(output.suffix+'.tmp')
    if pending.exists():
        raise FileExistsError(f'Incomplete index exists: {pending}')
    connection=sqlite3.connect(pending)
    try:
        connection.execute('PRAGMA journal_mode=OFF')
        connection.execute('PRAGMA synchronous=OFF')
        connection.execute('CREATE TABLE vendors (entity_id TEXT UNIQUE, business_name TEXT, '
                           'business_address TEXT, country TEXT, name_norm TEXT, address_norm TEXT)')
        batch=[]; count=0
        for path in paths:
            for row in read_rows(path):
                batch.append((row['entity_id'],row['business_name'],row['business_address'],
                              row['country'],normalize(row['business_name']),normalize(row['business_address'])))
                if len(batch)>=batch_size:
                    connection.executemany('INSERT INTO vendors VALUES (?,?,?,?,?,?)',batch)
                    count+=len(batch);batch.clear();connection.commit()
                    if count%1000000==0: print(f'Indexed {count:,} vendor records',flush=True)
        if batch:
            connection.executemany('INSERT INTO vendors VALUES (?,?,?,?,?,?)',batch)
            count+=len(batch);connection.commit()
        connection.execute('CREATE INDEX vendor_name_exact ON vendors(country,name_norm)')
        connection.execute('CREATE INDEX vendor_address_exact ON vendors(country,address_norm)')
        connection.execute("CREATE VIRTUAL TABLE vendor_name USING fts5(name_norm, "
                           "content='vendors', content_rowid='rowid', tokenize='trigram')")
        connection.execute("CREATE VIRTUAL TABLE vendor_address USING fts5(address_norm, "
                           "content='vendors', content_rowid='rowid', tokenize='trigram')")
        connection.execute("INSERT INTO vendor_name(vendor_name) VALUES('rebuild')")
        connection.execute("INSERT INTO vendor_address(vendor_address) VALUES('rebuild')")
        connection.commit()
        connection.execute('PRAGMA optimize')
        connection.commit()
        connection.close()
        pending.replace(output)
        print(f'Built {output}: {count:,} records',flush=True)
    except BaseException:
        connection.close()
        raise


def lookup(connection,reference,per_channel=80):
    """Return bounded union of exact, token, and 4-character substring hits."""
    country=reference['country']
    results={}
    fields='v.entity_id,v.business_name,v.business_address,v.country'
    def add(cursor):
        for mid,name,address,region in cursor:
            results[mid]={'entity_id':mid,'business_name':name,
                          'business_address':address,'country':region}
    for col,value in (('name_norm',reference['name']),('address_norm',reference['address'])):
        if value:
            add(connection.execute(f'SELECT {fields} FROM vendors v WHERE v.country=? AND v.{col}=? LIMIT ?',
                                   (country,value,per_channel)))
    for table,text in (('vendor_name',reference['name']),('vendor_address',reference['address'])):
        tokens=sorted({t for t in text.split() if len(t)>=4 and t.isalpha()},
                      key=lambda t:(-len(t),t))[:3]
        # Phrase queries on the local trigram index recover substrings and joined words.
        for token in tokens:
            phrase='"'+token.replace('"','""')+'"'
            add(connection.execute(f'SELECT {fields} FROM {table} f JOIN vendors v ON v.rowid=f.rowid '
                                   f'WHERE {table} MATCH ? AND v.country=? '
                                   f'ORDER BY bm25({table}) LIMIT ?',
                                   (phrase,country,per_channel)))
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('student_resource/dataset'))
    p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    build([args.data/args.split/f'{args.split}_source{s}.tsv' for s in (2,3)],args.output)

if __name__=='__main__': main()
