"""Independent tests for deterministic synthetic benchmark construction; no downloads."""
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_benchmark',ROOT/'scripts/prepare_benchmark.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
sys.path.insert(0,str(ROOT/'python'))
from reference import normalize,count_bruteforce


class BenchmarkDataTests(unittest.TestCase):
    def test_small_graphs_against_independent_enumeration(self):
        cases=[]
        for c,k in itertools.product([1,2,3],[2,3,5]): cases.append(('clique_chain',{'cliques':c,'size':k}))
        for r,c in itertools.product([1,2,3],[1,2,4]): cases.append(('grid',{'rows':r,'cols':c}))
        for g,k in itertools.product([1,2,4],[1,2,4]): cases.append(('shared_hub',{'groups':g,'size':k}))
        for a,b in itertools.product([0,1,2],[0,1,2]):
            if a+b: cases.append(('mixed_cliques',{'small_count':a,'small_size':3,'large_count':b,'large_size':5}))
        for family,p in cases:
            with self.subTest(family=family,parameters=p):
                records=list(prepare.synthetic_edges(family,p));nodes,edges=normalize(records)
                expected=prepare.synthetic_spec(family,p)
                self.assertEqual((len(nodes),len(edges),count_bruteforce(records)),tuple(expected[k] for k in ['vertices','undirected_edges','triangles']))
                self.assertEqual(len(records),len(edges)+int(family=='grid' and p['rows']*p['cols']==1))
                self.assertTrue(all(u<v for u,v in records if u!=v))

    def test_catalog_course_counts_and_mixed_budget(self):
        rows={r['id']:r for r in prepare.load_catalog()['datasets']}
        def stats(id):
            r=rows[id];return prepare.synthetic_spec(r['family'],r['parameters'])
        self.assertEqual(stats('S1'),dict(vertices=16384,undirected_edges=254463,triangles=2539520))
        self.assertEqual(stats('S2'),dict(vertices=1048576,undirected_edges=2095104,triangles=0))
        self.assertEqual(stats('S3'),dict(vertices=262145,undirected_edges=655360,triangles=655360))
        self.assertEqual(stats('S6')['undirected_edges'],stats('S7')['undirected_edges'])
        self.assertLess(stats('S6')['triangles'],stats('S7')['triangles'])
        self.assertEqual(len(prepare.selected_rows(prepare.load_catalog(),'core',None)),14)

    def test_reproducible_streamed_file_and_no_overwrite(self):
        row=dict(id='tiny',family='shared_hub',parameters={'groups':3,'size':4})
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.txt',Path(d)/'b.txt'
            ma,mb=prepare.generate(row,a),prepare.generate(row,b)
            self.assertEqual(a.read_bytes(),b.read_bytes());self.assertEqual(ma,mb)
            self.assertEqual(ma['sha256'],hashlib.sha256(a.read_bytes()).hexdigest())
            records=[tuple(map(int,line.split())) for line in a.read_text().splitlines() if not line.startswith('#')]
            self.assertEqual(count_bruteforce(records),30)
            with self.assertRaises(FileExistsError): prepare.generate(row,a)

    def test_gzip_crc_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            archive,target=Path(d)/'input.gz',Path(d)/'output.txt'
            content=b'# test\n0 1\n1 2\n0 2\n'
            archive.write_bytes(gzip.compress(content));prepare.unpack_gzip(archive,target)
            self.assertEqual(target.read_bytes(),content)
            with self.assertRaises(FileExistsError): prepare.unpack_gzip(archive,target)
            target.unlink();archive.write_bytes(gzip.compress(content)[:-5])
            with self.assertRaises((EOFError,OSError)): prepare.unpack_gzip(archive,target)
            self.assertFalse(target.exists());self.assertFalse(target.with_suffix('.txt.unpack.part').exists())

    def test_selection_and_paths_exclude_catalog_text(self):
        catalog=prepare.load_catalog()
        with self.assertRaises(ValueError):prepare.selected_rows(catalog,'core',['bogus'])
        with self.assertRaises(ValueError):prepare.selected_rows(catalog,'core',['S1','S1'])
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'generated').mkdir();(Path(d)/'generated'/'S1.txt').write_text('0 1\n')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_benchmark.py'),'paths','--ids','S1','--data-dir',d],capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout.strip(),str((Path(d)/'generated'/'S1.txt').resolve()))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_benchmark.py'),'paths','--ids','S2','--data-dir',d],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertEqual(result.stdout,'')


if __name__=='__main__': unittest.main()
