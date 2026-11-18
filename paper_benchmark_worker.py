"""Execute train/OBB test inside one isolated source tree's Python import path.

This program MUST be launched with cwd/PYTHONPATH set to either the proposal
source or the official ETFNet clone. It refuses accidental cross-imports.
"""
from __future__ import annotations
import argparse,json,os,sys,hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('train','test'),required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--project',required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--imgsz',type=int,required=True)
    p.add_argument('--device',default='0')
    p.add_argument('--resume',default='')
    p.add_argument('--metrics-path',default='')
    a=p.parse_args()
    import ultralytics
    imported=Path(ultralytics.__file__).resolve();source=Path(a.source).resolve()
    if not imported.is_relative_to(source):
        raise RuntimeError('FOREIGN package import: '+str(imported)+'; expected '+str(source))
    from ultralytics import YOLO
    model_config=Path(a.model).resolve()
    if a.phase=='train':
        if a.resume:
            ckpt=Path(a.resume)
            if not ckpt.is_file():raise FileNotFoundError(ckpt)
            model=YOLO(str(ckpt),task='obb')
            model.train(resume=True)
        else:
            if not model_config.is_file():raise FileNotFoundError(model_config)
            model=YOLO(str(model_config),task='obb')
            if model.task!='obb':raise RuntimeError('Not an oriented detector')
            model.train(data=a.data,project=a.project,name=a.name,exist_ok=True,
                epochs=a.epochs,batch=a.batch,imgsz=a.imgsz,device=a.device,workers=0,
                seed=a.seed,deterministic=True,optimizer='SGD',lr0=0.01,lrf=0.01,
                momentum=0.937,weight_decay=0.0005,plots=False,
                # Training is under the original author architecture, not local substitutes.
                ch=6)
        return
    checkpoint=Path(a.model).resolve()
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    model=YOLO(str(checkpoint),task='obb')
    val=model.val(data=a.data,split='test',project=a.project,name=a.name,exist_ok=True,
                  imgsz=a.imgsz,batch=a.batch,device=a.device,workers=0,
                  save_json=True,save_txt=False,plots=False,conf=0.001,iou=0.70,
                  max_det=300,ch=6)
    project_path=Path(a.project)/a.name
    pred=project_path/'predictions.json'
    # The validator itself must persist even an empty list. Missing JSON
    # means validation did not supply trustworthy prediction evidence.
    if not pred.is_file():
        raise RuntimeError('Validator failed to produce predictions.json; refusing fabricated empty predictions')
    validated_predictions=json.loads(pred.read_text())
    if not isinstance(validated_predictions,list):
        raise RuntimeError('No valid OBB predictions.json list from validator')
    report={'status':'completed','predictions':str(pred),
            'predictions_sha256':hashlib.sha256(pred.read_bytes()).hexdigest(),
            'n_predictions':len(validated_predictions),'imported_ultralytics':str(imported),
            'save_dir':str(project_path),'split':'test','imgsz':a.imgsz,'seed':a.seed,
            'postprocessing':{'conf':0.001,'iou':0.70,'max_det':300,'obb':True},
            'validator_stats':getattr(val,'results_dict',{}),
            'test_metrics_for_diagnostics_only':True}
    target=Path(a.metrics_path) if a.metrics_path else project_path/'worker_receipt.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,default=float)+'\n')
    print(json.dumps(report,indent=2,default=float))

if __name__=='__main__':main()
