"""Common *oriented polygon IoU* AP101 evaluator for RGB/NIR model predictions.

Both author ETFNet and proposed MR-SJPA provide native Ultralytics OBB
`predictions.json` containing image_id, category_id, score, poly. This module
recomputes BOTH mAP values from the identical test labels, IoU, classes,
matching and AP implementation (as opposed to mixing upstream metric logs).
"""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np,yaml
from paper_benchmark_dataset import digest

THRESHOLDS=[round(0.5+0.05*i,2) for i in range(10)]
EVALUATOR='common_obb_polygon_iou_AP101_v1'
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}


def polygon(coords):
    if len(coords)!=8:raise ValueError('Expected 4 vertices, 8 values')
    a=np.asarray(coords,dtype=np.float32).reshape(4,2)
    if not np.isfinite(a).all():raise ValueError('Nonfinite polygon')
    area=float(cv2.contourArea(a))
    if area<=1e-9:raise ValueError('Degenerate oriented box')
    # OpenCV's convex intersection expects contours with consistent point order.
    if not cv2.isContourConvex(a):raise ValueError('Nonconvex or disordered OBB polygon')
    return a


def iou(a,b):
    aa=float(cv2.contourArea(a));bb=float(cv2.contourArea(b))
    if aa<=0 or bb<=0:return 0.
    inter,_=cv2.intersectConvexConvex(a,b)
    return max(0.,min(1.,float(inter)/(aa+bb-float(inter)))) if aa+bb>float(inter) else 0.


def ap101(dets,gt,threshold):
    n=sum(len(v) for v in gt.values())
    if not n:return None
    matched=defaultdict(set);tp=[];fp=[]
    for im,confidence,vertices in sorted(dets,key=lambda r:(-r[1],r[0])):
        choices=gt.get(im,[]);best=-1;best_iou=threshold-1e-10
        for j,target in enumerate(choices):
            if j in matched[im]:continue
            ov=iou(vertices,target)
            if ov>=threshold and ov>best_iou:best,best_iou=j,ov
        tp.append(1 if best>=0 else 0);fp.append(0 if best>=0 else 1)
        if best>=0:matched[im].add(best)
    t=f=0;precision=[];recall=[]
    for a,b in zip(tp,fp):
        t+=a;f+=b;precision.append(t/(t+f));recall.append(t/n)
    return sum(max((p for p,r in zip(precision,recall) if r>=i/100-1e-12),default=0.)
               for i in range(101))/101.


def load_test(data_yaml):
    data_yaml=Path(data_yaml).resolve();data=yaml.safe_load(data_yaml.read_text())
    root=Path(data.get('path',data_yaml.parent))
    if not root.is_absolute():root=(data_yaml.parent/root).resolve()
    names=data['names']
    if isinstance(names,dict):names=[names[k] for k in sorted(names,key=lambda x:int(x))]
    if not names or len(names)!=len(set(names)):raise ValueError('Invalid class order')
    dirs=[root/data[k] for k in ('test','test_ir','test_labels')]
    if any(not d.is_dir() for d in dirs):raise FileNotFoundError('Missing paired held-out OBB test split')
    images=sorted(p for p in dirs[0].rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
    if not images:raise ValueError('Held-out test images empty')
    gt={};images_by_stem={};fingerprint=hashlib.sha256()
    for im in images:
        rel=im.relative_to(dirs[0]);key=im.stem
        if key in images_by_stem:raise ValueError('Duplicate test image stem '+key)
        ir=dirs[1]/rel;lab=dirs[2]/rel.with_suffix('.txt')
        if not ir.is_file() or not lab.is_file():raise FileNotFoundError('Missing test pair '+key)
        decoded=cv2.imread(str(im),cv2.IMREAD_UNCHANGED)
        if decoded is None:raise ValueError('Unreadable test image '+str(im))
        height,width=decoded.shape[:2]
        if cv2.imread(str(ir),cv2.IMREAD_UNCHANGED) is None:raise ValueError('Unreadable IR '+str(ir))
        images_by_stem[key]=(width,height)
        for path in (im,ir,lab):fingerprint.update((path.name+'\0'+digest(path)).encode())
        for line in lab.read_text().splitlines():
            if not line.strip():continue
            bits=line.split()
            if len(bits)!=9:raise ValueError('Expected native OBB labels, not HBB: '+str(lab))
            cls=int(bits[0]);raw=list(map(float,bits[1:]))
            if str(cls)!=bits[0] or not 0<=cls<len(names) or any(not 0<=v<=1 for v in raw):
                raise ValueError('Bad normalized OBB label '+str(lab))
            abs_box=[v*(width if i%2==0 else height) for i,v in enumerate(raw)]
            gt.setdefault((key,cls),[]).append(polygon(abs_box))
    return names,images_by_stem,gt,fingerprint.hexdigest(),data


def evaluate(data_yaml,pred_json,checkpoint,method,seed,source_commit,output):
    pred_json=Path(pred_json).resolve();checkpoint=Path(checkpoint).resolve()
    if not pred_json.is_file() or not checkpoint.is_file():raise FileNotFoundError('Missing predictions or checkpoint')
    names,images,gt,datadigest,data=load_test(data_yaml)
    prediction=json.loads(pred_json.read_text())
    if not isinstance(prediction,list):raise ValueError('Expected list of OBB predictions')
    bycls=defaultdict(list)
    for row in prediction:
        for key in ('image_id','category_id','score','poly'):
            if key not in row:raise ValueError('Missing prediction field '+key)
        name=str(row['image_id'])
        if name not in images:raise ValueError('Prediction outside test split: '+name)
        cls=int(row['category_id']);score=float(row['score'])
        if not 0<=cls<len(names) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('Invalid class or confidence')
        bycls[cls].append((name,score,polygon(row['poly'])))
    perclass={}
    for cls,name in enumerate(names):
        ref={im:poly for (im,c),poly in gt.items() if c==cls}
        if not ref:continue
        ap=[ap101(bycls[cls],ref,threshold) for threshold in THRESHOLDS]
        perclass[name]={'ap50':ap[0],'map50_95':sum(ap)/len(ap),'n_gt':sum(map(len,ref.values()))}
    if not perclass:raise ValueError('Cannot evaluate test data without any labelled objects')
    # Stable order-independent split fingerprint: names + paired paths, including
    # train/val membership, to catch leakage and accidental split differences.
    member_hash=hashlib.sha256()
    root=Path(data['path'])
    for split in ('train','val','test'):
        images_dir=root/data[split]
        for p in sorted(x for x in images_dir.rglob('*') if x.is_file() and x.suffix.lower() in EXTS):
            member_hash.update((split+'/'+p.relative_to(images_dir).as_posix()+'\n').encode())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    r={'status':'completed','method':method,'seed':int(seed),'source_commit':source_commit,
       'map50':sum(v['ap50'] for v in perclass.values())/len(perclass),
       'map50_95':sum(v['map50_95'] for v in perclass.values())/len(perclass),
       'per_class':perclass,'n_test_images':len(images),'n_test_gt':sum(v['n_gt'] for v in perclass.values()),
       'n_test_predictions':len(prediction), 'dataset_digest':datadigest,
       'split_digest':member_hash.hexdigest(),'class_order':list(names),
       'iou_thresholds':THRESHOLDS,'evaluator_id':EVALUATOR,
       'eval_geometry':'obb','eval_split':'test','polygon_iou':'OpenCV intersectConvexConvex',
       'metric_definition':'101-point interpolated AP; mean over GT-present classes; class-specific confidence sorting',
       'prediction_source':str(pred_json),'prediction_sha256':digest(pred_json),
       'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
       'warnings':['NMS is upstream Ultralytics OBB NMS; same flags required for both models',
                   'New VEDAI holdout is not ETFNet original paper benchmark']}
    output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--pred',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--method',required=True)
    p.add_argument('--seed',required=True,type=int);p.add_argument('--source-commit',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(evaluate(a.data,a.pred,a.checkpoint,a.method,a.seed,a.source_commit,a.out),indent=2))
