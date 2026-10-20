import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import yaml
from PIL import Image
from paper_common_hbb_eval import ap_101,evaluate_hbb,run,read_run_gt_and_predictions,THRESHOLDS
from official_upstream_compat import apply_icafusion_path_fix
from verified_fair_comparison import compare

class EvaluatorTests(unittest.TestCase):
    def dataset(self,root):
        root=Path(root)
        for split in ('train','val','test'):
            for ch in ('rgb','ir'):
                p=root/ch/'images'/split/(split+'.png');p.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(32,32),(75,75,75)).save(p)
            p=root/'labels'/split/(split+'.txt');p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('0 .5 .5 .5 .5\n')
        dat={'path':str(root),'train':'rgb/images/train','val':'rgb/images/val','test':'rgb/images/test',
             'test_ir':'ir/images/test','test_labels':'labels/test','names':['car']}
        y=root/'hbb.yaml';y.write_text(yaml.safe_dump(dat));return y
    def test_perfect_one_class_score(self):
        gt=[{'image_id':'test','class_id':0,'box':[.25,.25,.75,.75]}]
        pr=[{'image_id':'test','class_id':0,'box':[.25,.25,.75,.75],'score':.99}]
        r=evaluate_hbb(['car'],gt,pr)
        self.assertAlmostEqual(r['map50'],1);self.assertAlmostEqual(r['map50_95'],1)
    def test_duplicate_prediction_penalty(self):
        tar={'test':[[.25,.25,.75,.75]]}
        perfect=[('test',.99,[.25,.25,.75,.75])]
        self.assertAlmostEqual(ap_101(perfect,tar,.5),1.)
        fpearly=[('test',.99,[0,0,.1,.1]),('test',.8,[.25,.25,.75,.75])]
        self.assertLess(ap_101(fpearly,tar,.5),1.)
    def test_empty_prediction_requires_explicit_override(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);y=self.dataset(t/'source');p=t/'empty';p.mkdir()
            with self.assertRaisesRegex(ValueError,'No predictions'):read_run_gt_and_predictions(y,p)
            *_,g,s=read_run_gt_and_predictions(y,p,allow_zero=True)
            self.assertEqual(len(g),64);self.assertEqual(len(s),64)
    def test_real_prediction_creates_evidence_and_comparable_csv(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);y=self.dataset(t/'src')
            cp=t/'weights.pt';cp.write_bytes(b'fake weights for parser unit test, NOT a real model')
            pr1=t/'p1';pr1.mkdir();(pr1/'test.txt').write_text('0 0.5 0.5 0.5 0.5 0.99\n')
            pr2=t/'p2';pr2.mkdir();(pr2/'test.txt').write_text('0 0.5 0.5 0.5 0.5 0.88\n')
            a=run('icafusion_hbb',y,pr1,cp,t/'e1')
            b=run('cft_hbb',y,pr2,cp,t/'e2')
            self.assertAlmostEqual(a['map50'],1.)
            v=compare([t/'e1/icafusion_hbb_receipt.json',t/'e2/cft_hbb_receipt.json'],t/'result.csv')
            self.assertEqual(len(v['methods']),2)
            self.assertTrue((t/'result.csv').is_file())
    def test_no_out_of_split_or_bad_unpatched_icafusion(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);y=self.dataset(t/'src');pred=t/'pred';pred.mkdir()
            (pred/'train.txt').write_text('0 .5 .5 .5 .5 .8\n')
            with self.assertRaisesRegex(ValueError,'outside test'):read_run_gt_and_predictions(y,pred)
            (pred/'train.txt').unlink();(pred/'test.txt').write_text('37,5,5,5,5,0.8\n')
            with self.assertRaisesRegex(ValueError,'prediction format'):read_run_gt_and_predictions(y,pred)
    def test_superyolo_id_remapping(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);y=self.dataset(t/'src');p=t/'p';p.mkdir()
            (p/'00000007_co.txt').write_text('0 .5 .5 .5 .5 .75\n')
            a=read_run_gt_and_predictions(y,p,image_id_map={'00000007':'test'})
            self.assertEqual(a[2][0]['image_id'],'test')
    def test_icafusion_patch_change_only_export(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);repo=t/'upstream';repo.mkdir()
            src=r'''p = "/home/shen/Chenyf/FLIR-align-3class/labels/test"\ni = labels_list.index(str(path.stem) + '.txt')\nxywh = (xyxy2xywh2(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh\nline = (i+1, *xywh, conf) if save_conf else (i+1, *xywh)  # label format\nf.write(('%g,' * len(line)).rstrip(",") % line + '\\n')\n'''
            src=src.replace(chr(92)*2+'n',chr(92)+'n')
            (repo/'test.py').write_text(src)
            with patch('official_upstream_compat.verify_repo',return_value={'source_commit':'deadbeef'}):
                r=apply_icafusion_path_fix(repo)
            new=(repo/'test.py').read_text()
            self.assertIn('line = (cls,',new)
            self.assertIn(' / gn',new)
            self.assertIn('labels_list',new)
            self.assertEqual((repo/'test.py.orig').read_text(),src)
            self.assertEqual(r['status'],'I_O_patch_applied_NOT_model_validated')
            with patch('official_upstream_compat.verify_repo',return_value={'source_commit':'deadbeef'}):
                with self.assertRaisesRegex(ValueError,'anchor changed'):apply_icafusion_path_fix(repo)
if __name__=='__main__':unittest.main()
