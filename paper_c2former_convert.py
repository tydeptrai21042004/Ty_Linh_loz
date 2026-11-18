"""Convert actual MMRotate C2Former outputs to common OBB polygon predictions.

Supports author's MMRotate 0.3 style list[image][class] arrays with Nx6
(cx,cy,w,h,angle,score) and Nx9 (x1,y1,...,x4,y4,score).
Never manufactures detections. Pickle must be produced by trusted local code.
"""
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
import cv2,numpy as np


def convert(pkl_file,mapping,output,*,num_classes):
    ids=json.loads(Path(mapping).read_text())
    test=[r for r in ids if r['split']=='test']
    if not test:raise ValueError('No source test ID mapping')
    with Path(pkl_file).open('rb') as f:rows=pickle.load(f)
    if not isinstance(rows,list) or len(rows)!=len(test):
        raise ValueError(f'Output length {len(rows)} does not match C2Former test dataset {len(test)}')
    if num_classes<1:raise ValueError('No classes')
    out=[]
    for source,item in zip(test,rows):
        if isinstance(item,tuple):item=item[0]
        if not isinstance(item,(list,tuple)) or len(item)!=num_classes:
            raise ValueError('Unexpected MMRotate per-class prediction format')
        for cls,arr in enumerate(item):
            arr=np.asarray(arr,dtype=np.float64)
            if arr.size==0:continue
            if arr.ndim!=2 or arr.shape[1] not in (6,9) or not np.isfinite(arr).all():
                raise ValueError('Unsupported C2Former oriented bbox format')
            for row in arr:
                if arr.shape[1]==6:
                    cx,cy,w,h,angle,score=map(float,row)
                    if w<=0 or h<=0:raise ValueError('Invalid MMRotate width/height')
                    degrees=np.rad2deg(angle)
                    points=cv2.boxPoints(((cx,cy),(w,h),float(degrees))).reshape(-1).tolist()
                else:
                    points=list(map(float,row[:8]));score=float(row[8])
                if not 0<=score<=1:raise ValueError('Invalid score')
                out.append({'image_id':Path(source['source_rel']).stem,
                            'category_id':cls,'score':score,'poly':points})
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,indent=2)+'\n')
    return len(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pkl',required=True)
    p.add_argument('--map',required=True);p.add_argument('--output',required=True)
    p.add_argument('--classes',type=int,required=True)
    a=p.parse_args();print('Converted predictions:',convert(a.pkl,a.map,a.output,num_classes=a.classes))
