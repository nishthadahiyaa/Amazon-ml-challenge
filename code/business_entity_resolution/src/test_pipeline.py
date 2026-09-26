import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from preprocessing import normalize, prepare, blocking_keys, fold_latin
from features import features, FEATURE_NAMES
from blocking import retrieve
from blocking_v3 import retrieve as retrieve_v3
from pipeline import save_json, train, predict
from evaluation import entity_f05
from validate_submission import validate


def record(sid,name,address,country='US'):
    return {'entity_id':sid,'business_name':name,'business_address':address,'country':country}


def write_source(path,records):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['entity_id','business_name','business_address','country'],delimiter='\t')
        w.writeheader(); w.writerows(records)


class PipelineTests(unittest.TestCase):
    def test_unicode_and_missing_address(self):
        self.assertEqual(normalize('मॉडर्न फाइनेंस'),'मॉडर्न फाइनेंस')
        a=prepare(record('S1-a','École SARL',''))
        b=prepare(record('S2-b','École SARL',''))
        values=features(a,b)
        self.assertEqual(len(values),len(FEATURE_NAMES))
        self.assertEqual(values[FEATURE_NAMES.index('address_exact')],0)
        self.assertEqual(values[FEATURE_NAMES.index('address_missing')],1)

    def test_canonical_number_feature_handles_leading_zero(self):
        a=prepare(record('S1-a','Beth Center','0156 Bentley Parc'))
        b=prepare(record('S2-b','Beth Center','156 Bentley Parc'))
        values=features(a,b)
        self.assertEqual(values[FEATURE_NAMES.index('number_jaccard')],0)
        self.assertEqual(values[FEATURE_NAMES.index('canonical_number_jaccard')],1)

    def test_blocking_name_reordering_and_country(self):
        a=prepare(record('S1-a','Acme Robotics Incorporated','123 Main Street'))
        b=prepare(record('S2-b','Robotics Acme Inc','123 Main St'))
        self.assertTrue(set(blocking_keys(a))&set(blocking_keys(b)))
        b['country']='France'
        self.assertFalse(set(blocking_keys(a))&set(blocking_keys(b)))

    def test_address_block_recovers_cross_script_name(self):
        a=prepare(record('S1-a','Prime Finance','1052 Faridabad Haryana','India'))
        b=prepare(record('S2-b','प्राइम फाइनेंस','B3 1052 Faridabad Delhi Haryana','India'))
        shared=set(blocking_keys(a)) & set(blocking_keys(b))
        self.assertTrue(any(k[1]=='address_number' for k in shared))

    def test_latin_fold_preserves_indic_marks(self):
        self.assertEqual(fold_latin('école'),'ecole')
        self.assertEqual(fold_latin('फाइनेंस'),'फाइनेंस')

    def test_retrieval_caps_candidates_before_scoring(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vendors.tsv'
            write_source(path,[record('S2-close','Acme Robotics','99 Elm Rd'),
                               record('S2-best','Acme Robotics','12 Oak St'),
                               record('S2-wrong-country','Acme Robotics','12 Oak St','France')])
            refs={'S1-a':prepare(record('S1-a','Acme Robotics','12 Oak St'))}
            result=retrieve(refs,[path],top_k=1)
            self.assertEqual(set(result['S1-a']),{'S2-best'})

    def test_v3_address_channel_recovers_short_address_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vendors.tsv'
            refs={'S1-a':prepare(record('S1-a','Prime Finance',
                'Longwordone Longwordtwo Longwordthree Longwordfour Olive Plaza','India'))}
            write_source(path,[record('S2-match','प्राइम फाइनेंस','Olive Plaza','India')])
            old=retrieve(refs,[path],top_k=120,blocking_version=2)
            new=retrieve_v3(refs,[path],top_k=120)
            self.assertNotIn('S2-match',old['S1-a'])
            self.assertIn('S2-match',new['S1-a'])

    def test_end_to_end_output_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); test=root/'dataset/test'; test.mkdir(parents=True)
            refs={}; truth={}; candidates={}; vendor_rows=[]; ref_rows=[]
            for i in range(200):
                sid=f'S1-{i}'; mid=f'S2-{i}'
                row=record(sid,f'Business {i} Inc',f'{i+1} Oak Rd')
                ref_rows.append(row); refs[sid]=prepare(row)
                good=record(mid,f'Business {i} Inc',f'{i+1} Oak Rd')
                bad=record(f'S3-{i}',f'Business {i} Inc',f'{i+9000} Other Avenue')
                vendor_rows.append(good)
                truth[sid]=[mid] if i%10 else []
                candidates[sid]={mid:prepare(good),bad['entity_id']:prepare(bad)}
            save_json(root/'artifacts/training_candidates.json',{'refs':refs,'truth':truth,'candidates':candidates,
                      'config':{'top_k':5,'max_block':200,'seed':2026,'sample_size':200,'version':1}})
            args=Namespace(artifacts=root/'artifacts',reports=root/'reports',threads=1,
                           data=root/'dataset',output=root/'output',reference_batch_size=100)
            train(args)
            # Includes an unseen country and a reference with no possible candidates.
            ref_rows=ref_rows[:3]+[record('S1-fr','École Unique','63 Rue Neuve','France'),record('S1-empty','No Match','Unknown','France')]
            vendor_rows=vendor_rows[:3]+[record('S2-fr','École Unique','63 Rue Neuve','France')]
            write_source(test/'test_source1.tsv',ref_rows)
            write_source(test/'test_source2.tsv',vendor_rows)
            write_source(test/'test_source3.tsv',[])
            predict(args)
            errors,warnings=validate(str(args.output/'matching_results.tsv'),str(args.output/'candidate_pairs.tsv'),str(test),check_ids=True)
            self.assertEqual(errors,[]); self.assertEqual(warnings,[])
            with (args.output/'candidate_pairs.tsv').open() as f:
                output={r['source1_entity_id']:r['candidate_entity_ids'] for r in csv.DictReader(f,delimiter='\t')}
            self.assertIn('S2-fr',output['S1-fr']); self.assertEqual(output['S1-empty'],'')

if __name__=='__main__':
    unittest.main()
