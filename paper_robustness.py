"""Deterministic test-only RGB/IR corruption datasets and common-OBB robustness tables.

20 corruption conditions plus clean. Never modifies train/val, boxes or source
images. All models see identical corrupted bytes per dataset/condition.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import cv2
import numpy as np
import yaml


def conditions():
    result=[{'id':'clean','modality':'none','kind':'clean','severity':0}]
    for kind in ('gaussian_noise','gaussian_blur','darkness','occlusion'):
        for modality in ('rgb','ir'):
            for severity in (1,2):
                result.append({'id':f'{kind}_{modality}_{severity}','modality':modality,
                               'kind':kind,'severity':severity})
    for severity in (1,2):
        result.append({'id':f'shift_ir_{severity}','modality':'ir','kind':'shift','severity':severity})
    for modality in ('rgb','ir'):
        result.append({'id':f'dropout_{modality}','modality':modality,'kind':'dropout','severity':1})
    assert len(result)==21
    return result


def corrupt(image, entry, seed):
    image=np.asarray(image)
    if entry['kind']=='clean':return image.copy()
    rng=np.random.default_rng(seed)
    kind=entry['kind'];severity=entry['severity']
    if kind=='gaussian_noise':
        sigma=8 if severity==1 else 20
        return np.clip(image.astype(np.float32)+rng.normal(0,sigma,image.shape),0,255).astype(np.uint8)
    if kind=='gaussian_blur':
        n=3 if severity==1 else 7
        return cv2.GaussianBlur(image,(n,n),0)
    if kind=='darkness':
        return np.clip(image.astype(np.float32)*(0.7 if severity==1 else 0.45),0,255).astype(np.uint8)
    if kind=='occlusion':
        h,w=image.shape[:2];frac=0.10 if severity==1 else 0.25
        hh=max(1,int(h*math.sqrt(frac)));ww=max(1,int(w*math.sqrt(frac)))
        y=int(rng.integers(0,h-hh+1));x=int(rng.integers(0,w-ww+1))
        out=image.copy();out[y:y+hh,x:x+ww]=0;return out
    if kind=='shift':
        h,w=image.shape[:2]
        return cv2.warpAffine(image,np.float32([[1,0,severity],[0,1,severity]]),
                              (w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    if kind=='dropout':return np.zeros_like(image)
    raise ValueError('Unknown corruption: '+kind)


def make_view(source_yaml,destination,condition):
    source_yaml=Path(source_yaml).resolve();destination=Path(destination).resolve()
    data=yaml.safe_load(source_yaml.read_text())
    base=Path(data.get('path',source_yaml.parent))
    if not base.is_absolute():base=(source_yaml.parent/base).resolve()
    for field in ('test','test_ir','test_labels'):
        if field not in data:raise ValueError('No heldout test field: '+field)
    rgb_dir=base/data['test'];ir_dir=base/data['test_ir'];lab_dir=base/data['test_labels']
    if not rgb_dir.is_dir() or not ir_dir.is_dir() or not lab_dir.is_dir():raise FileNotFoundError('Missing original heldout folders')
    output_yaml=destination/'data.yaml';manifest=destination/'corruption_manifest.json'
    # Strictly avoid reusing a stale result from a different source/split.
    source_digest=hashlib.sha256(source_yaml.read_bytes()).hexdigest()
    if output_yaml.exists():
        old=json.loads(manifest.read_text())
        if old['source_yaml_sha256']!=source_digest or old['condition']!=condition:
            raise RuntimeError('Corruption cache hash mismatch; choose a new output dir')
        return output_yaml
    image_extensions={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
    files=sorted(p for p in rgb_dir.rglob('*') if p.is_file() and p.suffix.lower() in image_extensions)
    if not files:raise ValueError('No test images')
    for rgb in files:
        rel=rgb.relative_to(rgb_dir);ir=ir_dir/rel;label=lab_dir/rel.with_suffix('.txt')
        if not ir.is_file() or not label.is_file():raise FileNotFoundError('Missing RGB/IR/label: '+str(rel))
        raw_rgb=cv2.imread(str(rgb),cv2.IMREAD_UNCHANGED);raw_ir=cv2.imread(str(ir),cv2.IMREAD_UNCHANGED)
        if raw_rgb is None or raw_ir is None or raw_rgb.shape[:2]!=raw_ir.shape[:2]:
            raise ValueError('Unreadable or unpaired test input '+str(rel))
        rng_seed=int.from_bytes(hashlib.sha256((condition['id']+'|'+rel.as_posix()).encode()).digest()[:8],'big')
        rgb_modified=corrupt(raw_rgb,condition,rng_seed) if condition['modality']=='rgb' else raw_rgb
        ir_modified=corrupt(raw_ir,condition,rng_seed) if condition['modality']=='ir' else raw_ir
        for source,directory in ((rgb_modified,destination/'rgb/images/test'),(ir_modified,destination/'ir/images/test')):
            target=directory/rel.with_suffix('.png');target.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(target),source):raise RuntimeError('Cannot encode test image '+str(target))
    data['path']=str(base)
    data['test']=str(destination/'rgb/images/test')
    data['test_ir']=str(destination/'ir/images/test')
    data['test_labels']=str(lab_dir)
    data['robustness_condition']=condition
    destination.mkdir(parents=True,exist_ok=True)
    output_yaml.write_text(yaml.safe_dump(data,sort_keys=False))
    manifest.write_text(json.dumps({'source_yaml':str(source_yaml),'source_yaml_sha256':source_digest,
                                    'condition':condition,'n_test_images':len(files)},indent=2)+'\n')
    return output_yaml


def aggregate(rows,out):
    """Aggregate measured, evaluator-matched seed receipts by condition.

    All condition records must contain three seeds for each selected method.
    Return one mean ± SD per method, plus delta against that method's clean score.
    """
    from paper_benchmark_stats import _stats
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    groups={};contracts={}
    for condition,method,receipt in rows:
        if receipt.get('status')!='completed' or receipt.get('eval_split')!='test':
            raise ValueError('Incomplete corruption evidence')
        name=receipt['seed'];key=(condition,method)
        groups.setdefault(key,{})
        if name in groups[key]:raise ValueError('Duplicate robustness seed')
        groups[key][name]=receipt
        contract=(receipt['dataset_digest'],receipt['split_digest'],tuple(receipt['class_order']),
                  receipt['evaluator_id'],tuple(receipt['iou_thresholds']))
        if condition in contracts and contracts[condition]!=contract:
            raise ValueError('Models did not share exactly the same corrupted dataset and evaluator')
        contracts[condition]=contract
    summarized=[]
    for (condition,method),perseed in sorted(groups.items()):
        if set(perseed)!={0,1,2}:raise ValueError('Robustness needs three seeds for '+str((condition,method)))
        result={'condition':condition,'method':method}
        for metric in ('map50','map50_95'):
            res=_stats([perseed[s][metric] for s in (0,1,2)])
            result[metric+'_mean']=res['mean'];result[metric+'_sd']=res['sd']
        summarized.append(result)
    clean={(r['method']):r for r in summarized if r['condition']=='clean'}
    if not clean:raise ValueError('Missing clean evaluation')
    for r in summarized:
        if r['method'] not in clean:raise ValueError('Missing matched clean method')
        r['delta_map50_95_from_clean']=r['map50_95_mean']-clean[r['method']]['map50_95_mean']
    with (out/'robustness_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summarized[0]));w.writeheader();w.writerows(summarized)
    (out/'robustness_summary.json').write_text(json.dumps({'results':summarized,'n_conditions':len(contracts),
        'note':'Corruptions are deterministic, TEST-only and identical for each model. 3-seed CI is descriptive.'},indent=2)+'\n')
    return summarized


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--condition',default='all',choices=['all']+[x['id'] for x in conditions()])
    a=p.parse_args()
    for condition in conditions():
        if a.condition not in ('all',condition['id']):continue
        print(condition['id'], make_view(a.data,Path(a.out)/condition['id'],condition))
