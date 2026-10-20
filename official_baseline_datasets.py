#!/usr/bin/env python3
"""Audited dataset adapters for upstream C2Former, SuperYOLO, ICAFusion and CFT.

No model reimplementation, no results fabrication. Import this module standalone
from the repository root. Source is an SJPA-style YOLO OBB data YAML with paired
RGB/IR train/val/test directory fields. Outputs are reproducible file views.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from pathlib import Path

from external_paper_baselines import parse_names, convert_obb_line_to_hbb, prepare_hbb, sha256

EXTS = {'.png','.bmp','.jpg','.jpeg','.tif','.tiff'}


def _root(yaml_file: Path, data: dict) -> Path:
    p = Path(data.get('path', yaml_file.parent))
    return (p if p.is_absolute() else yaml_file.parent / p).resolve()


def _items(source_yaml):
    from PIL import Image
    source_yaml = Path(source_yaml).resolve()
    data, names = parse_names(source_yaml)
    root = _root(source_yaml, data)
    inventory, seen = [], {}
    for split in ('train','val','test'):
        if split not in data:
            continue
        paths = []
        for field in (split, split+'_ir', split+'_labels'):
            if field not in data: raise ValueError('Missing dataset YAML key '+field)
            p = (root / data[field]).resolve()
            if not p.is_dir(): raise FileNotFoundError(p)
            paths.append(p)
        rgb_root,ir_root,lab_root = paths
        pairs = sorted(p for p in rgb_root.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
        if not pairs: raise ValueError('Empty split '+split)
        seen[split] = set()
        for rgb in pairs:
            rel = rgb.relative_to(rgb_root)
            stem = rel.with_suffix('').as_posix()
            if stem in seen[split]: raise ValueError('Duplicate RGB stem '+stem)
            seen[split].add(stem)
            ir, label = ir_root/rel, lab_root/rel.with_suffix('.txt')
            if not ir.is_file() or not label.is_file(): raise FileNotFoundError('Missing paired IR or annotation for '+str(rgb))
            with Image.open(rgb) as im: size = im.size
            with Image.open(ir) as im: ir_size = im.size
            if size != ir_size: raise ValueError('RGB/IR image sizes differ for '+str(rgb))
            lines = [s.strip() for s in label.read_text(encoding='utf-8').splitlines() if s.strip()]
            for line in lines: convert_obb_line_to_hbb(line,len(names))
            inventory.append((split,rel,rgb,ir,label,lines,size))
    if not {'train','val'}.issubset(seen): raise ValueError('Require nonempty train and val splits')
    for a in seen:
        for b in seen:
            if a < b and seen[a]&seen[b]: raise ValueError('Overlap by source stem between '+a+' and '+b)
    # Guard against per-split cross-modal or annotation extra/missing files.
    for split in seen:
        pass
    return names, inventory


def _ready_empty(out):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('Output must be empty: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    return out


def _image(src,dst, format='JPEG'):
    from PIL import Image
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        if format == 'JPEG': img.convert('RGB').save(dst, 'JPEG', quality=100, subsampling=0)
        else: img.convert('RGB').save(dst,'PNG')


def _dota(line,names,size):
    values = line.split()
    n = int(values[0]); nums = [float(x) for x in values[1:]]
    w,h = size
    return ' '.join([*(f'{a*(w if i%2==0 else h):.6f}' for i,a in enumerate(nums)), names[n], '0'])


def c2former_dota_view(source_yaml: Path, output: Path):
    """Exact filenames required by upstream DroneVehicleDataset.load_annotations.

    It globs *_tir.txt and pairs <id>.jpg with <id>_tir.jpg. Arbitrary
    source IDs are remapped to numeric IDs (its loader splits on '_').
    JPEG transcoding is recorded; use SAME normalized image view in cross-method
    comparisons or report this source of protocol difference.
    """
    source_yaml=Path(source_yaml).resolve(); out=Path(output).resolve()
    names,items=_items(source_yaml)
    if source_yaml.parent==out or source_yaml.parent in out.parents: raise ValueError('Output cannot be below dataset root')
    if len(set(names))!=len(names) or any(' ' in name for name in names): raise ValueError('Class names must be unique DOTA-safe tokens')
    out=_ready_empty(out)
    id_map=[]; count={}
    for idx,(split,rel,rgb,ir,lab,lines,size) in enumerate(items,1):
        key=f'{idx:08d}'; count[split]=count.get(split,0)+1
        images=out/split/'images'; ann=out/split/'annfiles'
        _image(rgb,images/f'{key}.jpg'); _image(ir,images/f'{key}_tir.jpg')
        ann.mkdir(parents=True,exist_ok=True)
        (ann/f'{key}_tir.txt').write_text('\n'.join(_dota(l,names,size) for l in lines)+('\n' if lines else ''),encoding='utf-8')
        id_map.append({'split':split,'id':key,'source_rel':rel.as_posix(),'width':size[0],'height':size[1]})
    (out/'source_id_map.json').write_text(json.dumps(id_map,indent=2)+'\n')
    report={'status':'prepared_not_trained','adapter':'official_DroneVehicleDataset',
            'geometry':'obb','classes':names,'count':count,
            'dataset_sha256':sha256(source_yaml),'jpeg_transcoded':True,
            'paired_layout':'<split>/images/<id>.jpg, <id>_tir.jpg; <split>/annfiles/<id>_tir.txt',
            'limitation':'JPEG transcodes input; use shared decoded pixels to compare across models.'}
    (out/'c2former_dota_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def make_c2former_config(repo:Path, view:Path, names:list[str], *, output:Path|None=None):
    """Generate a MMRotate child config, preserving the original C2Former backbone.

    MMRotate config inheritance updates dataset roots, class list, and 2 detection
    head class counts. Does not edit author backbone/fusion weights.
    """
    repo=Path(repo).resolve(); view=Path(view).resolve()
    official=repo/'configs/s2anet/s2anet_c2former_fpn_1x_dota_le135.py'
    if not official.is_file() or not (repo/'mmrotate/datasets/dronevehicle.py').is_file():
        raise FileNotFoundError('Not a full official C2Former checkout')
    if not (view/'c2former_dota_manifest.json').is_file(): raise FileNotFoundError('Prepare C2 view first')
    if not names or any(' ' in x or not x for x in names): raise ValueError('Invalid class names')
    output=Path(output or repo/'configs/s2anet/sjpa_c2former_paired_local.py').resolve()
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    r=["# AUTO-GENERATED dataset/class adapter. Original network architecture retained.",
       "_base_ = ["+repr(str(official))+"]",
       'classes = '+repr(tuple(names)),
       'model = dict(fam_head=dict(num_classes='+str(len(names))+'), odm_head=dict(num_classes='+str(len(names))+'))',
       'data = dict(']
    for split in ('train','val','test'):
        curr=split if (view/split/'images').is_dir() else 'val'
        r.append('    '+split+'=dict(type="DroneVehicleDataset", classes=classes, '
                 +'ann_file='+repr(str(view/curr/'annfiles'))+', img_prefix='+repr(str(view/curr/'images'))+'),')
    r.append(')')
    output.write_text('\n'.join(r)+'\n',encoding='utf-8')
    return output


def legacy_hbb_view(hbb_yaml: Path, *, validation_as_test:bool=False):
    """Author YOLOv5 paired label views; labels/test ALWAYS maps to VAL.

    The ICAFusion author train.py uses this hardcoded directory while validating.
    Never copy held-out test annotations into labels/test during training.
    """
    import yaml
    hbb_yaml=Path(hbb_yaml).resolve(); data,names=parse_names(hbb_yaml)
    root=_root(hbb_yaml,data)
    if data.get('source',{}).get('geometry')!='hbb': raise ValueError('First convert OBB to HBB using prepare-hbb')
    counts={}; all_stems=set()
    for split in ('train','val','test'):
        if split not in data: continue
        src_lbl=root/data[split+'_labels']; folders=[root/data[split],root/data[split+'_ir']]
        if not src_lbl.is_dir(): raise FileNotFoundError(src_lbl)
        names_found=None
        for folder in folders:
            if not folder.is_dir(): raise FileNotFoundError(folder)
            if '/images/' not in str(folder).replace('\\','/')+'/': raise ValueError('YOLOv5 requires an images path component')
            dst=Path(str(folder).replace('/images/','/labels/'))
            files=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
            stems={p.stem for p in files}
            if len(stems)!=len(files): raise ValueError('Duplicate stems in '+str(folder))
            if names_found is not None and stems!=names_found: raise ValueError('RGB/IR stems differ')
            names_found=stems
            for img in files:
                if img.parent!=folder: raise ValueError('Legacy author evaluation requires FLAT unique filenames per split; flatten source first')
                s=src_lbl/(img.stem+'.txt')
                if not s.is_file(): raise FileNotFoundError(s)
                dst.mkdir(parents=True,exist_ok=True)
                target=dst/(img.stem+'.txt')
                if target.exists() and target.read_bytes()!=s.read_bytes(): raise FileExistsError(target)
                if not target.exists(): shutil.copy2(s,target)
        counts[split]=len(names_found)
    # ICAFusion training & test.py both index filenames from path/labels/test.
    # This is intentionally the validation split, *not heldout* test.
    # Keep a SEPARATE directory; root/labels/test is the actual heldout GT!
    dest=root/'icafusion_training_view'/'labels/test'; dest.mkdir(parents=True,exist_ok=True)
    val_lbl=root/data['val_labels']
    for p in val_lbl.glob('*.txt'):
        target=dest/p.name
        if target.exists() and target.read_bytes()!=p.read_bytes(): raise FileExistsError(target)
        if not target.exists(): shutil.copy2(p,target)
    vals={p.stem for p in (root/data['val']).glob('*') if p.suffix.lower() in EXTS}
    if vals!={p.stem for p in dest.glob('*.txt')}: raise ValueError('ICAFusion val label index is not identical to VAL filenames')
    cfg={'path':str(root/'icafusion_training_view'),'train_rgb':str(root/data['train']),'train_ir':str(root/data['train_ir']),
         'val_rgb':str(root/data['val']),'val_ir':str(root/data['val_ir']),
         'train':str(root/data['train']),'val':str(root/data['val']),
         'nc':len(names),'names':names}
    path=root/'data_legacy_rgb_ir.yaml'; path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    report={'status':'prepared_not_trained','output_yaml':str(path),'split_counts':counts,
            'labels_test_meaning':'validation_only_for_ICAFusion_training','val_class_names':names}
    (root/'legacy_upstream_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def make_icafusion_heldout_view(hbb_yaml:Path, output:Path):
    """Creates a separate YAML+labels/test index for ICAFusion heldout eval."""
    import yaml
    hbb_yaml=Path(hbb_yaml).resolve(); data,names=parse_names(hbb_yaml)
    root=_root(hbb_yaml,data)
    if 'test' not in data: raise ValueError('No heldout test split')
    out=_ready_empty(output)
    val_src=root/data['test_labels']; labels=out/'labels/test'; labels.mkdir(parents=True)
    imgs=[p for p in (root/data['test']).iterdir() if p.suffix.lower() in EXTS]
    if not imgs: raise ValueError('Empty test split')
    for im in imgs:
        src=val_src/(im.stem+'.txt')
        if not src.is_file(): raise FileNotFoundError(src)
        shutil.copy2(src, labels/src.name)
    cfg={'path':str(out),'train_rgb':str(root/data['train']),'train_ir':str(root/data['train_ir']),
         'val_rgb':str(root/data['test']),'val_ir':str(root/data['test_ir']),
         'train':str(root/data['train']),'val':str(root/data['test']),
         'nc':len(names),'names':names}
    p=out/'data_icafusion_heldout.yaml'; p.write_text(yaml.safe_dump(cfg,sort_keys=False))
    return {'output_yaml':str(p),'split':'test','num_images':len(imgs),'status':'heldout_view_not_evaluated'}


def superyolo_view(low_hbb_yaml:Path, high_hbb_yaml:Path, output:Path):
    """SuperYOLO official requires paired 512/1024 inputs from same acquisitions.

    Deliberately rejects fake 1024 created from 512. The user must provide
    independently captured/source-preserved 1024 RGB/NIR frames. We cannot
    prove authenticity from pixels, so the manifest explicitly says so.
    """
    import yaml
    low_yaml=Path(low_hbb_yaml).resolve();high_yaml=Path(high_hbb_yaml).resolve()
    dl,names=parse_names(low_yaml); dh,nh=parse_names(high_yaml)
    if names!=nh: raise ValueError('High/low class orders differ')
    rlow=_root(low_yaml,dl); rhigh=_root(high_yaml,dh)
    out=_ready_empty(output)
    manifest=[]; counts={}; keyset=set()
    for split in ('train','val','test'):
        if split not in dl: continue
        if split not in dh: raise ValueError('High resolution missing '+split)
        folders=[rlow/dl[split],rlow/dl[split+'_ir'],rhigh/dh[split],rhigh/dh[split+'_ir']]
        for p in folders:
            if not p.is_dir(): raise FileNotFoundError(p)
        lowimgs=sorted(p for p in folders[0].rglob('*') if p.suffix.lower() in EXTS)
        his={p.relative_to(folders[2]).with_suffix('').as_posix() for p in folders[2].rglob('*') if p.suffix.lower() in EXTS}
        lows={p.relative_to(folders[0]).with_suffix('').as_posix() for p in lowimgs}
        if lows!=his: raise ValueError('High/low filenames differ for '+split)
        for rgb in lowimgs:
            rel=rgb.relative_to(folders[0]); hi_candidates=[p for p in folders[2].glob(str(rel.with_suffix('.*'))) if p.suffix.lower() in EXTS]
            if len(hi_candidates)!=1: raise ValueError('Missing unique matching high image '+str(rel))
            hi=hi_candidates[0]; ir=folders[1]/rel
            hi_ir_candidates=[p for p in folders[3].glob(str(rel.with_suffix('.*'))) if p.suffix.lower() in EXTS]
            if not ir.is_file() or len(hi_ir_candidates)!=1: raise FileNotFoundError('Paired IR missing '+str(rel))
            hir=hi_ir_candidates[0]
            from PIL import Image
            with Image.open(rgb) as im: low_size=im.size
            with Image.open(hi) as im: hi_size=im.size
            if low_size!=(512,512) or hi_size!=(1024,1024): raise ValueError('SuperYOLO SR data requires genuine 512x512/1024x1024 pairs; got '+str((low_size,hi_size)))
            ltext=(rlow/dl[split+'_labels']/rel.with_suffix('.txt')).read_text().strip().splitlines()
            htext=(rhigh/dh[split+'_labels']/rel.with_suffix('.txt')).read_text().strip().splitlines()
            if len(ltext)!=len(htext):raise ValueError('High/low annotations count mismatch '+str(rel))
            for x,y in zip(ltext,htext):
                xv=[float(v) for v in x.split()];yv=[float(v) for v in y.split()]
                if len(xv)!=5 or len(yv)!=5 or any(abs(a-b)>0.001 for a,b in zip(xv,yv)):
                    raise ValueError('High/low HBB annotation mismatch '+str(rel))
            key=f'{len(manifest)+1:08d}'
            for tier,images in (('VEDAI',(rgb,ir)),('VEDAI_1024',(hi,hir))):
                dest=out/tier/'images'
                _image(images[0],dest/(key+'_co.png'),'PNG')
                _image(images[1],dest/(key+'_ir.png'),'PNG')
                labels=out/tier/'labels';labels.mkdir(parents=True,exist_ok=True)
                (labels/(key+'.txt')).write_text('\n'.join(ltext)+('\n' if ltext else ''))
            manifest.append({'split':split,'id':key,'original':rel.as_posix()})
            counts[split]=counts.get(split,0)+1
    for split in counts:
        (out/(split+'_ids.txt')).write_text('\n'.join(str(out/('VEDAI_1024' if split=='train' else 'VEDAI')/'images'/r['id']) for r in manifest if r['split']==split)+'\n')
    cfg={'train':str(out/'train_ids.txt'),'val':str(out/'val_ids.txt'),
         'test':str(out/'test_ids.txt') if 'test' in counts else str(out/'val_ids.txt'),
         'nc':len(names),'names':names}
    p=out/'SRvedai.yaml';p.write_text(yaml.safe_dump(cfg,sort_keys=False))
    (out/'id_mapping.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report={'status':'prepared_not_trained','geometry':'hbb','output_yaml':str(p),'counts':counts,
            'requires_compat_patch':True,'train_list_1024_val_test_lists_512':True,'true_high_res_unverifiable':'must attest source acquisition, NOT upsampling',
            'low_root':str(out/'VEDAI'),'high_root':str(out/'VEDAI_1024')}
    (out/'superyolo_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    pa=argparse.ArgumentParser(description=__doc__); sub=pa.add_subparsers(dest='mode',required=True)
    for name in ('c2former','legacy','icafusion-test','superyolo'):
        p=sub.add_parser(name);p.add_argument('--output',type=Path)
        if name=='c2former':p.add_argument('--obb-data',type=Path,required=True)
        if name in ('legacy','icafusion-test'):p.add_argument('--hbb-data',type=Path,required=True)
        if name=='superyolo':
            p.add_argument('--low-hbb',type=Path,required=True);p.add_argument('--high-hbb',type=Path,required=True)
    args=pa.parse_args()
    if args.mode=='c2former':result=c2former_dota_view(args.obb_data,args.output)
    elif args.mode=='legacy':result=legacy_hbb_view(args.hbb_data)
    elif args.mode=='icafusion-test':result=make_icafusion_heldout_view(args.hbb_data,args.output)
    else: result=superyolo_view(args.low_hbb,args.high_hbb,args.output)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
