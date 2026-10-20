"""Offline tests, grounded in inspected upstream loader naming/CLI contracts."""
import json,tempfile,unittest
from pathlib import Path

from official_baseline_datasets import (c2former_dota_view,legacy_hbb_view,
                                       make_icafusion_heldout_view,superyolo_view,make_c2former_config)
from official_baseline_runs import command_for,validate_data_yaml
from external_paper_baselines import prepare_hbb
from verified_fair_comparison import compare,sha256


class AdapterTests(unittest.TestCase):
    def make_obb(self,base,wh=(512,512),bad_ir=False):
        import yaml
        from PIL import Image
        base=Path(base);base.mkdir(parents=True,exist_ok=True)
        for split in ('train','val','test'):
            for kind in ('rgb','ir'):
                file=base/kind/'images'/split/(split+'.png')
                file.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB' if kind=='rgb' else 'L', (wh[0]//2,wh[1]) if bad_ir and kind=='ir' else wh,30).save(file)
            ann=base/'labels'/split/(split+'.txt');ann.parent.mkdir(parents=True,exist_ok=True)
            ann.write_text('0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n')
        conf={'path':str(base),'names':['vehicle'],'source':{'task':'obb'}}
        for split in ('train','val','test'):
            conf.update({split:f'rgb/images/{split}',split+'_ir':f'ir/images/{split}',split+'_labels':f'labels/{split}'})
        p=base/'data.yaml';p.write_text(yaml.safe_dump(conf));return p

    def test_c2former_real_author_filenames_and_nondefault_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);src=self.make_obb(t/'original',wh=(50,30))
            res=c2former_dota_view(src,t/'paired_dota')
            self.assertEqual(res['count'],{'train':1,'val':1,'test':1})
            self.assertTrue((t/'paired_dota/train/images/00000001.jpg').is_file())
            self.assertTrue((t/'paired_dota/train/images/00000001_tir.jpg').is_file())
            ann=(t/'paired_dota/train/annfiles/00000001_tir.txt').read_text().split()
            self.assertEqual(ann[-2:],['vehicle','0'])
            self.assertAlmostEqual(float(ann[0]),12.5)
            repo=t/'official';model=repo/'configs/s2anet/s2anet_c2former_fpn_1x_dota_le135.py'
            model.parent.mkdir(parents=True);model.write_text('model=dict()')
            (repo/'mmrotate/datasets').mkdir(parents=True);(repo/'mmrotate/datasets/dronevehicle.py').write_text('')
            config=make_c2former_config(repo,t/'paired_dota',['vehicle'])
            content=config.read_text()
            self.assertIn('num_classes=1',content)
            self.assertIn('00000001', (t/'paired_dota/source_id_map.json').read_text())

    def test_legacy_val_alias_never_copies_test_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);src=self.make_obb(t/'original')
            prepare_hbb(src,t/'hbb');data=t/'hbb/data_hbb.yaml'
            out=legacy_hbb_view(data)
            self.assertTrue((t/'hbb/icafusion_training_view/labels/test/val.txt').is_file())
            self.assertFalse((t/'hbb/icafusion_training_view/labels/test/test.txt').is_file())
            for k in ('cft_hbb','icafusion_hbb'):
                self.assertEqual(validate_data_yaml(k,Path(out['output_yaml']))['nc'],1)
            v=make_icafusion_heldout_view(data,t/'heldout')
            self.assertTrue((t/'heldout/labels/test/test.txt').is_file())
            self.assertFalse((t/'heldout/labels/test/val.txt').exists())
            validate_data_yaml('icafusion_hbb',v['output_yaml'])

    def test_superyolo_true_multiresolution_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);slo=self.make_obb(t/'low')
            shi=self.make_obb(t/'high',wh=(1024,1024))
            prepare_hbb(slo,t/'low_hbb');prepare_hbb(shi,t/'high_hbb')
            res=superyolo_view(t/'low_hbb/data_hbb.yaml',t/'high_hbb/data_hbb.yaml',t/'sup')
            self.assertEqual(res['counts'],{'train':1,'val':1,'test':1})
            self.assertTrue((t/'sup/VEDAI/images/00000001_co.png').is_file())
            self.assertTrue((t/'sup/VEDAI/images/00000001_ir.png').is_file())
            self.assertTrue((t/'sup/VEDAI_1024/images/00000001_co.png').is_file())
            self.assertTrue((t/'sup/VEDAI_1024/labels/00000001.txt').is_file())
            cfg=validate_data_yaml('superyolo_hbb',res['output_yaml'])
            self.assertEqual(cfg['nc'],1)
            self.assertIn('VEDAI_1024',Path(cfg['train']).read_text())
            self.assertNotIn('VEDAI_1024',Path(cfg['val']).read_text())
            with self.assertRaisesRegex(ValueError,'SuperYOLO requires'):
                command_for('superyolo_hbb',python='python',data='data',model='m',out='x')
            cmd=command_for('superyolo_hbb',python='python',data='data',model='m',out='x',super_resolution=True)
            for arg in ('--super','--hr_input','--ch','64','RGB+IR+MF'):self.assertIn(arg,cmd)
            test_cmd=command_for('superyolo_hbb',python='python',data='data',model='m',out='x',phase='test',checkpoint='best.pt')
            self.assertIn('test',test_cmd[test_cmd.index('--task')+1:test_cmd.index('--task')+2])

    def test_mismatched_ir_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);src=self.make_obb(t/'original',bad_ir=True)
            with self.assertRaisesRegex(ValueError,'sizes differ'):
                c2former_dota_view(src,t/'out')

    def test_cft_not_generic_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);src=self.make_obb(t/'source')
            prepare_hbb(src,t/'hbb');r=legacy_hbb_view(t/'hbb/data_hbb.yaml')
            wrong=t/'model_transformer.yaml';wrong.write_text('backbone: Transformer')
            with self.assertRaisesRegex(ValueError,'transformerx3'):
                validate_data_yaml('cft_hbb',r['output_yaml'],wrong)
            right=t/'paper_transformerx3.yaml';right.write_text('backbone: Transformer')
            validate_data_yaml('cft_hbb',r['output_yaml'],right)

    def test_comparison_contract_mixing_geometry_and_fake_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);artifact=root/'test.bin';artifact.write_bytes(b'good evidence')
            base={'status':'completed','source_kind':'measured','dataset_digest':'d','split_digest':'s',
                  'class_order':['car'],'evaluator_id':'same-common-eval-v1','iou_thresholds':[0.5,0.75],
                  'input_resolution':[512,512],'evaluation_split':'test',
                  'postprocess_protocol':'nms 0.6','metric_definition':'mAP',
                  'map50':0.5,'map50_95':0.3}
            for key in ('checkpoint','predictions','ground_truth'):
                base[key]=str(artifact);base[key+'_sha256']=sha256(artifact)
            a=dict(base,method='sjpa_tr_proposal',geometry='obb');b=dict(base,method='c2former_obb',geometry='obb')
            paths=[]
            for i,r in enumerate((a,b)):
                path=root/(str(i)+'.json');path.write_text(json.dumps(r));paths.append(path)
            v=compare(paths,root/'good.csv');self.assertEqual(len(v['methods']),2)
            b['geometry']='hbb';paths[1].write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError,'UNFAIR'):compare(paths,root/'bad.csv')
            b['geometry']='obb';b['predictions_sha256']='garbage';paths[1].write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError,'sha256'):compare(paths,root/'bad.csv')

if __name__=='__main__':unittest.main()
