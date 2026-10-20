#!/usr/bin/env python3
"""Small, audited I/O-only patches on cloned ORIGINAL author repositories.

Never patches model/fusion layers. The original file is preserved as .orig and
SHA256 before/after recorded. All anchors MUST match or patch refuses to run.
"""
from __future__ import annotations
import argparse,json,re,shutil
from pathlib import Path
from external_paper_baselines import sha256,verify_repo

ICAFUSION_BAD_PATH = 'p = "/home/shen/Chenyf/FLIR-align-3class/labels/test"'
ICAFUSION_DATASET_PATH = ('p = os.path.join(yaml.safe_load(Path(opt.data).read_text(encoding="utf-8"))'
                          '["path"], "labels", "test")')


def _patch(repo:Path,key:str,filename:str,replacements:tuple[tuple[str,str],...]):
    repo=Path(repo).resolve();info=verify_repo(key,repo)
    p=repo/filename;old=p.read_text(encoding='utf-8');new=old
    for before,after in replacements:
        if new.count(before)!=1: raise ValueError(f'Official {key} source anchor changed: {before[:100]}')
        new=new.replace(before,after,1)
    if new==old:raise ValueError('Empty patch')
    backup=p.with_name(p.name+'.orig')
    if backup.exists(): raise FileExistsError('Already patched: '+str(backup))
    shutil.copy2(p,backup);p.write_text(new,encoding='utf-8')
    result={'status':'I_O_patch_applied_NOT_model_validated','method':key,'upstream_commit':info['source_commit'],
            'file':str(p),'sha256_original':sha256(backup),'sha256_patched':sha256(p),'backup':str(backup)}
    receipt=repo.parent/(key+'.io_patch.json');receipt.write_text(json.dumps(result,indent=2)+'\n')
    return result


def apply_icafusion_path_fix(repo):
    # AUTHOR test.py uses labels/test regardless of val/test split. Adapter must
    # put the selected EVALUATION split here. Its train.py also assumes this.
    # Standalone author's exported TXT is NOT YOLO: first column is image index
    # rather than predicted class, coordinates are in pixels, and CSV commas
    # are used. Convert ONLY output format so common evaluation can read it.
    # The model, NMS and AP logic are unchanged.
    export_replacements = (
        (ICAFUSION_BAD_PATH, ICAFUSION_DATASET_PATH),
        ("i = labels_list.index(str(path.stem) + '.txt')",
         "assert (str(path.stem) + '.txt') in labels_list, 'Prediction image missing from selected evaluation split'"),
        ("xywh = (xyxy2xywh2(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh",
         "xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh"),
        ("line = (i+1, *xywh, conf) if save_conf else (i+1, *xywh)  # label format",
         "line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # real predicted class"),
        ('f.write((\'%g,\' * len(line)).rstrip(",") % line + \'\\n\')',
         "f.write(('%g ' * len(line)).rstrip() % line + '\\n')"),
    )
    return _patch(repo,'icafusion_hbb','test.py',export_replacements)


def apply_superyolo_path_fix(repo, data_root):
    """Fix author-local absolute dirs and unsafe substring IR filenames.

    Full SuperYOLO retains original SR and MF branch. Dataset *must* contain
    VEDAI/images/*.png and VEDAI_1024/images/*.png at the supplied root.
    """
    root=Path(data_root).resolve()
    for suffix in ('VEDAI/images','VEDAI/labels','VEDAI_1024/images','VEDAI_1024/labels'):
        if not (root/suffix).is_dir(): raise FileNotFoundError(root/suffix)
    r=repr(str(root))
    source=Path(repo)/'utils/datasets.py'
    text=source.read_text(encoding='utf-8')
    before1="self.img_path = '/home/data/zhangjiaqing/dataset/VEDAI/images/'"
    before2="self.img_path = '/home/data/zhangjiaqing/dataset/VEDAI_1024/images/'"
    # input IDs list with no extension; author appends _co.png. Modifying only
    # source directory declarations, not preprocessing or training algorithm.
    repl=((before1,'self.img_path = '+repr(str(root/'VEDAI/images')+'/')),
          (before2,'self.img_path = '+repr(str(root/'VEDAI_1024/images')+'/')),
          ("return [x.replace('co', 'ir') for x in img_paths]",
           "return [x[:-7] + '_ir.png' if x.endswith('_co.png') else x for x in img_paths]"))
    return _patch(repo,'superyolo_hbb','utils/datasets.py',repl)


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['icafusion_hbb','superyolo_hbb'],required=True)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--data-root',type=Path)
    args=p.parse_args()
    if args.method=='icafusion_hbb':report=apply_icafusion_path_fix(args.repo)
    else:
        if not args.data_root: p.error('--data-root required for SuperYOLO')
        report=apply_superyolo_path_fix(args.repo,args.data_root)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
