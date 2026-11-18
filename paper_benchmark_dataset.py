"""Deterministic, non-overlapping held-out test view of paired RGB/NIR VEDAI.

This is a NEW cross-dataset split, not an 'official VEDAI test' or a
reproduction of the original ETFNet paper numbers. Existing source validation
and any source test images are never moved into the train split.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil
from pathlib import Path
import yaml

EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def prepare(source_yaml, destination, *, test_fraction=0.2, split_seed=991, uniform_jpeg=True):
    source_yaml=Path(source_yaml).resolve(); destination=Path(destination).resolve()
    data=yaml.safe_load(source_yaml.read_text())
    base=Path(data.get('path') or source_yaml.parent)
    if not base.is_absolute():base=(source_yaml.parent/base).resolve()
    if not 0.05<=test_fraction<=0.45:raise ValueError('test_fraction must be 0.05..0.45')
    if base==destination or base in destination.parents or destination in base.parents:
        raise ValueError('Source and split destinations must not nest inside one another')
    required=('train','val','train_ir','val_ir','train_labels','val_labels')
    if any(k not in data for k in required):raise ValueError('Dataset must supply paired RGB, IR, and labels for train/val')
    if destination.exists() and (destination/'heldout_manifest.json').exists():
        old=json.loads((destination/'heldout_manifest.json').read_text())
        if (old.get('source_yaml_sha256')!=digest(source_yaml) or
            old.get('seed')!=split_seed or old.get('test_fraction')!=test_fraction or old.get('uniform_jpeg')!=uniform_jpeg):
            raise ValueError('Existing heldout output has different split provenance; use a different destination')
        return destination/'data.yaml'
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Destination must be empty')
    source_items={}; seen=set()
    for split in ('train','val','test'):
        if split not in data:continue
        keys=(split,split+'_ir',split+'_labels')
        roots=[(base/data[k]).resolve() for k in keys]
        if any(not x.is_dir() for x in roots):raise FileNotFoundError(f'Missing {split} paired roots')
        records=[]
        imgs=sorted(p for p in roots[0].rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
        if not imgs:raise ValueError('Empty source split: '+split)
        for rgb in imgs:
            rel=rgb.relative_to(roots[0]);key=rel.with_suffix('').as_posix().lower()
            if key in seen:raise ValueError(f'Split leakage / reused image stem: {key}')
            seen.add(key)
            ir=roots[1]/rel;lab=roots[2]/rel.with_suffix('.txt')
            if not ir.is_file() or not lab.is_file():raise FileNotFoundError(f'Unpaired data for {rgb}')
            records.append((key,rel,rgb,ir,lab))
        source_items[split]=records
    # Protect the existing independent test if the source provides it; otherwise
    # reserve a deterministic portion of TRAIN (never the validation images).
    heldout_from_train='test' not in source_items
    if heldout_from_train:
        pairs=source_items['train']
        if len(pairs)<15:raise ValueError('Need >=15 source training pairs for a defensible held-out split')
        sorted_pairs=sorted(pairs,key=lambda p:hashlib.sha256(f'{split_seed}:{p[0]}'.encode()).hexdigest())
        size=max(1,min(len(pairs)-1,round(len(pairs)*test_fraction)))
        source_items['train']=sorted_pairs[size:]
        source_items['test']=sorted_pairs[:size]
    counts={}; entries=[]
    for split in ('train','val','test'):
        group=source_items[split];counts[split]=len(group)
        if not group:raise ValueError('Empty split '+split)
        for key,rel,rgb,ir,lab in group:
            image_rel=rel.with_suffix('.jpg') if uniform_jpeg else rel
            targets=[destination/'rgb/images'/split/image_rel,
                     destination/'ir/images'/split/image_rel,
                     destination/'labels'/split/rel.with_suffix('.txt')]
            hashes=[]
            for src,target in zip((rgb,ir,lab),targets):
                target.parent.mkdir(parents=True,exist_ok=True)
                if uniform_jpeg and len(hashes)<2:
                    from PIL import Image
                    with Image.open(src) as img:img.convert('RGB').save(target,'JPEG',quality=100,subsampling=0)
                else:
                    try:os.link(src,target)
                    except OSError:shutil.copy2(src,target)
                hashes.append(digest(target))
            entries.append({'split':split,'key':key,'sha256_rgb':hashes[0],
                            'sha256_ir':hashes[1],'sha256_label':hashes[2]})
    config={k:v for k,v in data.items() if k not in ('path','train','val','test','train_ir','val_ir','test_ir','train_labels','val_labels','test_labels')}
    config['path']=str(destination)
    for split in ('train','val','test'):
        config.update({split:f'rgb/images/{split}',split+'_ir':f'ir/images/{split}',
                       split+'_labels':f'labels/{split}'})
    config['heldout_protocol']={'source':str(source_yaml),'source_sha256':digest(source_yaml),
                                'seed':split_seed,'train_reservation_fraction':test_fraction,
                                'source_test_preserved':not heldout_from_train,'uniform_jpeg_for_all_methods':uniform_jpeg,
                                'scope':'new cross-dataset split; not original published ETFNet score'}
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    manifest={'source_yaml_sha256':digest(source_yaml),'source_yaml':str(source_yaml),
              'seed':split_seed,'test_fraction':test_fraction,'uniform_jpeg':uniform_jpeg,'counts':counts,'files':entries,
              'test_origin':'source_test' if not heldout_from_train else 'reserved_source_train'}
    (destination/'heldout_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return destination/'data.yaml'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--test-fraction',type=float,default=0.2);p.add_argument('--seed',type=int,default=991)
    p.add_argument('--no-uniform-jpeg',action='store_true')
    a=p.parse_args();print(prepare(a.data,a.out,test_fraction=a.test_fraction,split_seed=a.seed,uniform_jpeg=not a.no_uniform_jpeg))
