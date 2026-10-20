#!/usr/bin/env python3
"""Shared, deterministic HBB AP evaluator for held-out paired RGB/IR data.

Accepts ORIGINAL YOLO prediction TXT (class xc yc w h confidence) saved by
SuperYOLO, CFT, and ICAFusion after I/O compatibility patch. This is a
101-recall-point AP implementation at 10 IoU thresholds, NOT official COCOeval.
For fair paper tables, run every detector on the same held-out images and use
one evaluator. No scores are borrowed from papers or generated without files.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from external_paper_baselines import parse_names, sha256

THRESHOLDS = [round(.5 + i*.05, 2) for i in range(10)]
EVALUATOR_ID = 'sjpa_shared_hbb_AP101_v1'

def _full_files(yaml_path):
    data, names = parse_names(yaml_path)
    root=Path(data.get('path',Path(yaml_path).parent))
    if not root.is_absolute(): root=Path(yaml_path).resolve().parent/root
    root=root.resolve()
    for key in ('train','val','test','test_ir','test_labels'):
        if key not in data:raise ValueError('Missing '+key+' in full HBB dataset')
    files={}
    for split in ('train','val','test'):
        p=root/data[split]
        if not p.is_dir():raise FileNotFoundError(p)
        images=[x for x in p.rglob('*') if x.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')]
        if not images:raise ValueError('Empty source split '+split)
        keys={x.stem for x in images}
        if len(keys)!=len(images):raise ValueError('Ambiguous image stem in '+split)
        files[split]=(p,images)
    if any(set(x.stem for x in files[a][1])&set(x.stem for x in files[b][1]) for a,b in (('train','val'),('train','test'),('val','test'))):
        raise ValueError('Split leakage: repeated source image ID')
    return root,data,names,files

def xywh_to_xyxy(values):
    xc,yc,w,h=values
    if not all(math.isfinite(v) for v in values) or w<=0 or h<=0 or not all(0<=v<=1 for v in values):
        raise ValueError('Invalid normalized HBB box: '+str(values))
    return [xc-w/2,yc-h/2,xc+w/2,yc+h/2]

def iou(a,b):
    x=max(0.,min(a[2],b[2])-max(a[0],b[0]))
    y=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    intersection=x*y
    aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]);bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return intersection/(aa+bb-intersection) if aa+bb>intersection else 0.

def ap_101(detections, gt_by_image, threshold):
    """Class-specific detections: (image_id, confidence, box)."""
    n_gt=sum(len(x) for x in gt_by_image.values())
    if not n_gt:return None
    used={im:set() for im in gt_by_image}
    tps=[];fps=[]
    for img,conf,box in sorted(detections,key=lambda x:(-x[1],x[0],*x[2])):
        targets=gt_by_image.get(img,[]);match=-1;best=threshold-1e-12
        for j,target in enumerate(targets):
            if j in used.get(img,set()):continue
            score=iou(box,target)
            if score>=threshold and score>best: match=j;best=score
        tp=int(match!=-1)
        if tp:used[img].add(match)
        tps.append(tp);fps.append(1-tp)
    cum_t=0;cum_f=0;prec=[];rec=[]
    for tp,fp in zip(tps,fps):
        cum_t+=tp;cum_f+=fp;rec.append(cum_t/n_gt);prec.append(cum_t/(cum_t+cum_f))
    return sum(max((p for p,r in zip(prec,rec) if r>=i/100-1e-12),default=0.) for i in range(101))/101

def read_run_gt_and_predictions(hbb_data,pred_dir,*,image_id_map=None,allow_zero=False):
    root,data,names,files=_full_files(hbb_data)
    from collections import defaultdict
    labels_path=root/data['test_labels']
    gt=[];prediction=[]
    test_stems={p.stem for p in files['test'][1]}
    if len(test_stems)!=len(files['test'][1]):raise ValueError('Duplicate test stem')
    paired_ir=root/data['test_ir']
    if not paired_ir.is_dir():raise FileNotFoundError(paired_ir)
    for img in files['test'][1]:
        im_key=img.stem
        if not (paired_ir/img.name).is_file():raise FileNotFoundError('Test paired IR absent for '+im_key)
        lab=labels_path/(im_key+'.txt')
        if not lab.is_file():raise FileNotFoundError(lab)
        for s in lab.read_text(encoding='utf-8').splitlines():
            if not s.strip():continue
            cols=s.split()
            if len(cols)!=5:raise ValueError('Not normalized HBB GT: '+str(lab))
            cls=int(cols[0])
            if str(cls)!=cols[0] or not 0<=cls<len(names):raise ValueError('Invalid GT class')
            gt.append({'image_id':im_key,'class_id':cls,'box':xywh_to_xyxy([float(x) for x in cols[1:]])})
    pred_dir=Path(pred_dir).resolve()
    if not pred_dir.is_dir():raise FileNotFoundError(pred_dir)
    filelist=sorted(pred_dir.glob('*.txt'))
    for p in filelist:
        if p.name=='result.txt':continue # ICAFusion also writes non-YOLO result.txt
        pred_stem=p.stem
        if image_id_map:
            # SuperYOLO filenames may have _co postfix; author output paths differ by branch.
            map_id=pred_stem[:-3] if pred_stem.endswith('_co') else pred_stem
            if map_id not in image_id_map:raise ValueError('Unknown predicted image ID '+pred_stem)
            stem=image_id_map[map_id]
        else:stem=pred_stem
        if stem not in test_stems:raise ValueError('Prediction contains image outside test: '+stem)
        for line in p.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            cols=line.split()
            if len(cols)!=6:raise ValueError('Not YOLO 6-column prediction format: '+str(p)+': '+line[:80])
            cls=int(cols[0]);v=[float(x) for x in cols[1:]]
            if not 0<=cls<len(names) or str(cls)!=cols[0]:
                # e.g. ICAFusion unpatched output contains image index rather than class
                raise ValueError('Predicted class out of range (ICAFusion export unpatched?): '+line)
            if not math.isfinite(v[-1]) or not 0<=v[-1]<=1:raise ValueError('Bad confidence '+line)
            prediction.append({'image_id':stem,'class_id':cls,'box':xywh_to_xyxy(v[:4]),'score':v[4]})
    if not prediction and not allow_zero:raise ValueError('No predictions found. Wrong output directory? --allow-zero only for proven empty detector')
    if not gt:raise ValueError('No held-out ground-truth instances; mAP undefined')
    # Path- and model-independent digest over annotation and image content in heldout split.
    dig=hashlib.sha256()
    for img in sorted(files['test'][1],key=lambda p:p.name):
        for path in (img,paired_ir/img.name,labels_path/(img.stem+'.txt')):
            dig.update(path.name.encode()+b'\0'+sha256(path).encode()+b'\0')
    dataset_digest=dig.hexdigest()
    # Train/val membership and GT annotation digests; distinct method model pipelines retain same split membership.
    sd=hashlib.sha256()
    for split in ('train','val','test'):
        for im in sorted(files[split][1],key=lambda p:p.name):
            sd.update((split+'/'+im.stem+'\n').encode())
    split_digest=sd.hexdigest()
    return names,gt,prediction,dataset_digest,split_digest

def evaluate_hbb(names,ground_truth,predictions):
    from collections import defaultdict
    ap={};present=[]
    for cls,name in enumerate(names):
        byim=defaultdict(list)
        for row in ground_truth:
            if row['class_id']==cls:byim[row['image_id']].append(row['box'])
        if not byim:
            ap[name]=None
            continue
        present.append(name)
        preds=[(r['image_id'],r['score'],r['box']) for r in predictions if r['class_id']==cls]
        scores=[ap_101(preds,byim,thr) for thr in THRESHOLDS]
        ap[name]={'AP50':scores[0],'AP50_95':sum(scores)/len(scores),'n_gt':sum(map(len,byim.values()))}
    if not present:raise ValueError('No GT class in test')
    return {'map50':sum(ap[c]['AP50'] for c in present)/len(present),
            'map50_95':sum(ap[c]['AP50_95'] for c in present)/len(present),
            'per_class':ap,'evaluated_classes':present,'unobserved_classes':[n for n in names if n not in present],
            'n_gt':len(ground_truth),'n_predictions':len(predictions)}

def run(method,hbb_yaml,pred_dir,checkpoint,out,*,mapping=None,input_size=512,nms='native-upstream',allow_zero=False):
    from verified_fair_comparison import KNOWN
    if method not in KNOWN:raise ValueError('Unknown method')
    if method=='c2former_obb':raise ValueError('C2Former oriented predictions require OBB evaluation, not HBB')
    if not Path(checkpoint).is_file():raise FileNotFoundError(checkpoint)
    id_map=None
    if mapping:
        items=json.loads(Path(mapping).read_text())
        id_map={x['id']:Path(x['original']).stem for x in items if x['split']=='test'}
        if len(id_map)!=len([x for x in items if x['split']=='test']):raise ValueError('Duplicate SuperYOLO mapping IDs')
    names,gt,predictions,dataset_digest,split_digest=read_run_gt_and_predictions(hbb_yaml,pred_dir,image_id_map=id_map,allow_zero=allow_zero)
    result=evaluate_hbb(names,gt,predictions)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    gt_path=out/(method+'_gt.json');pred_path=out/(method+'_predictions.json')
    if gt_path.exists() or pred_path.exists():raise FileExistsError('Refuse to overwrite evaluation evidence')
    gt_path.write_text(json.dumps(gt,indent=2)+'\n');pred_path.write_text(json.dumps(predictions,indent=2)+'\n')
    receipt={'method':method,'status':'completed','source_kind':'measured',
        'geometry':'hbb','dataset_digest':dataset_digest,'split_digest':split_digest,
        'class_order':names,'evaluator_id':EVALUATOR_ID,'iou_thresholds':THRESHOLDS,
        'input_resolution':[input_size,input_size],'evaluation_split':'test',
        'postprocess_protocol':nms,'metric_definition':'AP101 no crowd, no maxDets; mean over GT-present classes',
        'checkpoint':str(Path(checkpoint).resolve()),'checkpoint_sha256':sha256(checkpoint),
        'predictions':str(pred_path),'predictions_sha256':sha256(pred_path),
        'ground_truth':str(gt_path),'ground_truth_sha256':sha256(gt_path),**result,
        'limitations':'Common simple AP101, not official COCOeval; prediction NMS provenance must be inspected separately; comparison requires same dataset, geometry and split; SuperYOLO HR training is an extra input advantage.'}
    rpath=out/(method+'_receipt.json')
    if rpath.exists():raise FileExistsError('Refuse to overwrite result receipt')
    rpath.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True)
    p.add_argument('--hbb-data',type=Path,required=True)
    p.add_argument('--pred-dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--superyolo-id-map',type=Path)
    p.add_argument('--input-size',type=int,default=512)
    p.add_argument('--nms',default='native-upstream')
    p.add_argument('--allow-zero',action='store_true')
    a=p.parse_args()
    r=run(a.method,a.hbb_data,a.pred_dir,a.checkpoint,a.out,mapping=a.superyolo_id_map,
          input_size=a.input_size,nms=a.nms,allow_zero=a.allow_zero)
    print(json.dumps({k:r[k] for k in ('method','map50','map50_95','n_gt','n_predictions')},indent=2))
if __name__=='__main__':main()
