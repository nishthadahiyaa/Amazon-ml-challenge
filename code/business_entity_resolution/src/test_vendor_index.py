import tempfile
import unittest
from pathlib import Path
from vendor_index import build,lookup
from preprocessing import prepare
from test_pipeline import record,write_source
import sqlite3

class VendorIndexTests(unittest.TestCase):
    def test_persistent_local_search(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            vendors=path/'vendors.tsv'
            write_source(vendors,[record('S2-1','Acme Robotics','156 Bentley Parc'),
                                  record('S2-2','Elsewhere Bakery','99 Other Rd'),
                                  record('S2-3','Acme Robotics','156 Bentley Parc','France')])
            index=path/'vendors.sqlite'
            build([vendors],index,batch_size=2)
            with sqlite3.connect(index) as conn:
                hits=lookup(conn,prepare(record('S1-1','Acme Robotix','0156 Bentley Parc')))
            self.assertIn('S2-1',hits)
            self.assertNotIn('S2-3',hits)

if __name__=='__main__': unittest.main()
