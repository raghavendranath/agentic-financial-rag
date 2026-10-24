import importlib.util,sys,tempfile,unittest,zipfile,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class ReleaseTests(unittest.TestCase):
    def test_archive_paths_cannot_escape_or_write_unapproved_trees(self):
        m=module('unpack_artifacts')
        for p in ['../secret','/etc/passwd','experiments/x/data/../../../../evil','experiments/x/src/a.py','experiments\\x\\data\\evil']:
            with self.subTest(p=p),self.assertRaises(ValueError):m.member_path(p)
        p=m.member_path('experiments/x/data/inputs.json');self.assertEqual(p,ROOT/'experiments/x/data/inputs.json')
    def test_live_output_refuses_overwriting_published_inputs(self):
        m=module('live_run')
        with self.assertRaises(ValueError):m.output_path(str(ROOT/'experiments/new-run'))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):m.output_path(d)
    def test_invalid_or_unbounded_budget_refused_before_key_access(self):
        m=module('live_run')
        for b in ['nan','inf','0','-1','10000']:
            with self.subTest(b=b),self.assertRaises(SystemExit):m.arguments(['--benchmark','finqa','--out','work/new','--budget-usd',b])
    def test_result_summary_matches_declared_counts(self):module('release_check').check_results()
    def test_archive_install_refuses_changed_scientific_files(self):
        m=module('unpack_artifacts')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);member='experiments/x/data/inputs.json';raw=b'{}\n';asset=root/'a.zip'
            with zipfile.ZipFile(asset,'w') as z:z.writestr(member,raw)
            (root/'ARTIFACTS.json').write_text(json.dumps({'benchmark-artifacts.zip':{'bytes':asset.stat().st_size,'sha256':m.digest(asset)}}))
            (root/'SNAPSHOT-MANIFEST.json').write_text(json.dumps({member:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}))
            target=root/member;target.parent.mkdir(parents=True);target.write_text('changed')
            original=m.ROOT;m.ROOT=root
            try:
                # Pass root explicitly because member_path's default binds at import time.
                old=m.member_path;m.member_path=lambda name:old(name,root)
                with self.assertRaises(ValueError):m.install(asset)
            finally:m.ROOT=original
if __name__=='__main__':unittest.main()
