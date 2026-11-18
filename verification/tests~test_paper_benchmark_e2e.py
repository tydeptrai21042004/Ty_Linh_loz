from __future__ import annotations
import json
from pathlib import Path
import pytest


def fixture(tmp_path):
    import yaml
    from PIL import Image
    root=tmp_path/'original'
    for split,n in (('train',30),('val',6)):
        for i in range(n):
            key=f'{split}_{i:03d}.png'
            for mode in ('rgb','ir'):
                p=root/mode/'images'/split/key;p.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(32,32),(70,80,90) if mode=='rgb' else (30,40,50)).save(p)
            label=root/'labels'/split/f'{Path(key).stem}.txt'
            label.parent.mkdir(parents=True,exist_ok=True)
            label.write_text('0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n')
    spec={'path':str(root),'train':'rgb/images/train','val':'rgb/images/val',
          'train_ir':'ir/images/train','val_ir':'ir/images/val',
          'train_labels':'labels/train','val_labels':'labels/val','names':{0:'car'}}
    yaml_path=root/'data.yaml';yaml_path.write_text(yaml.safe_dump(spec))
    return yaml_path


def test_heldout_three_way_split_and_uniform_jpg(tmp_path):
    from paper_benchmark_dataset import prepare
    from paper_benchmark_eval import load_test
    source=fixture(tmp_path)
    result=prepare(source,tmp_path/'heldout',test_fraction=.2,split_seed=991)
    assert result.is_file()
    manifest=json.loads((result.parent/'heldout_manifest.json').read_text())
    assert manifest['counts']=={'train':24,'val':6,'test':6}
    assert len({row['key'] for row in manifest['files']})==36
    assert len(manifest['files'])==36
    assert all((result.parent/'rgb/images'/row['split']/(row['key']+'.jpg')).is_file() for row in manifest['files'])
    names,images,gt,_,_=load_test(result)
    assert names==['car'] and len(images)==6 and len(gt)==6
    assert prepare(source,result.parent)==result
    with pytest.raises(ValueError,match='different split'):
        prepare(source,result.parent,split_seed=9)


def test_common_oriented_evaluator_uses_pixels_and_test_only(tmp_path):
    from paper_benchmark_dataset import prepare
    from paper_benchmark_eval import evaluate
    src=fixture(tmp_path);data=prepare(src,tmp_path/'heldout')
    manifest=json.loads((data.parent/'heldout_manifest.json').read_text())
    rows=[]
    for item in manifest['files']:
        if item['split']=='test':
            rows.append({'image_id':Path(item['key']).stem,'category_id':0,'score':0.999,
                         'poly':[8,8,24,8,24,24,8,24]})
    p=tmp_path/'pred.json';p.write_text(json.dumps(rows))
    ckpt=tmp_path/'best.pt';ckpt.write_bytes(b'fixture-checkpoint-not-model')
    receipt=evaluate(data,p,ckpt,'mr_sjpa_proposal',0,'fixture_sha',tmp_path/'r.json')
    assert receipt['map50']==pytest.approx(1.)
    assert receipt['map50_95']==pytest.approx(1.)
    rows[0]['image_id']='validation_leakage'
    p.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='outside test'):
        evaluate(data,p,ckpt,'mr_sjpa_proposal',0,'fixture_sha',tmp_path/'bad.json')


def test_paired_statistics_no_fake_scores_and_fairness_gates(tmp_path):
    from paper_benchmark_stats import generate
    base={'status':'completed','evaluator_id':'shared_OBB_v1','dataset_digest':'abc',
          'split_digest':'def','class_order':['car'],'iou_thresholds':[0.5,0.55],
          'eval_geometry':'obb','eval_split':'test','checkpoint_sha256':'aa','source_commit':'source'}
    rows=[]
    for seed in range(3):
        for method,value in (('mr_sjpa_proposal',0.8),('etfnet_official_paper',0.7)):
            rows.append(dict(base,seed=seed,method=method,map50=value+0.01*seed,
                             map50_95=value-0.1+0.01*seed))
    stats=generate(rows,tmp_path/'stats')
    assert (tmp_path/'stats'/'paper_table.tex').is_file()
    assert 'sign_flip_exact_p_two_sided' in stats['paired']['mr_sjpa_proposal_minus_etfnet_official_paper']['map50']
    assert stats['paired']['mr_sjpa_proposal_minus_etfnet_official_paper']['map50']['sign_flip_exact_p_two_sided']==pytest.approx(.25)
    tampered=[x.copy() for x in rows];tampered[0]['eval_geometry']='hbb'
    with pytest.raises(ValueError,match='Unmatched'):
        generate(tampered,tmp_path/'unfair')
    with pytest.raises(ValueError,match='Missing'):
        generate(rows[:-1],tmp_path/'incomplete')


def test_c2former_mmrotate_pkl_conversion(tmp_path):
    import pickle,numpy as np
    from paper_c2former_convert import convert
    ids=[{'split':'test','source_rel':'train_002.jpg'}, {'split':'test','source_rel':'train_004.jpg'}]
    mp=tmp_path/'ids.json';mp.write_text(json.dumps(ids))
    prediction=[[np.array([[16,16,16,16,0.,.95]])],[np.zeros((0,6))]]
    pkl=tmp_path/'output.pkl'
    with pkl.open('wb') as f:pickle.dump(prediction,f)
    output=tmp_path/'pred.json'
    assert convert(pkl,mp,output,num_classes=1)==1
    assert json.loads(output.read_text())[0]['image_id']=='train_002'
