#!/usr/bin/env python3
"""Upstream-only launcher with reproducibility gates; no paper result claims.

Each external repository uses its own pinned legacy Python environment. This
launcher checks correct model/dataset adapter and writes command provenance.
--execute is opt-in and never implies correct trained weights or valid AP.
"""
from __future__ import annotations
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
from external_paper_baselines import get_method,load_registry,sha256,verify_repo


def command_for(key, *, python, data, model, out, epochs=100,batch=4,imgsz=512,
                weights='',device='0',super_resolution=False,phase='train',checkpoint=None):
    if epochs<1 or batch<1 or imgsz<32:raise ValueError('Bad epochs/batch/image size')
    if phase not in ('train','test'):raise ValueError('Bad phase')
    base=[str(python)]
    if key=='c2former_obb':
        if phase=='train':return base+['tools/train.py',str(model),'--work-dir',str(out)]
        if not checkpoint:raise ValueError('Testing requires checkpoint')
        return base+['tools/test.py',str(model),str(checkpoint),'--out',str(Path(out)/'predictions.pkl')]
    if key=='superyolo_hbb':
        if not super_resolution and phase=='train':
            raise ValueError('Full-paper SuperYOLO requires --super-resolution and independent HR data')
        if phase=='train':
            c=base+['train.py','--cfg',str(model),'--super','--hr_input',
                    '--train_img_size','1024','--test_img_size',str(imgsz),
                    '--data',str(data),'--ch','64','--input_mode','RGB+IR+MF',
                    '--epochs',str(epochs),'--batch-size',str(batch),'--device',str(device),
                    '--workers','2','--project',str(out),'--name',key,'--exist-ok']
            if weights:c+=['--weights',str(weights)]
            return c
        if not checkpoint:raise ValueError('Testing requires checkpoint')
        return base+['test.py','--weights',str(checkpoint),'--data',str(data),
                     '--task','test','--input_mode','RGB+IR+MF','--device',str(device),
                     '--img-size',str(imgsz),'--project',str(out),'--name',key,
                     '--exist-ok','--save-txt','--save-conf']
    if key in ('icafusion_hbb','cft_hbb'):
        if phase=='train':
            return base+['train.py','--cfg',str(model),'--data',str(data),
                         '--weights',str(weights),'--epochs',str(epochs),
                         '--batch-size',str(batch),'--img-size',str(imgsz),str(imgsz),
                         '--device',str(device),'--workers','2','--project',str(out),
                         '--name',key,'--exist-ok']
        if not checkpoint:raise ValueError('Testing requires checkpoint')
        return base+['test.py','--weights',str(checkpoint),'--data',str(data),
                     '--task','test','--img-size',str(imgsz),'--device',str(device),
                     '--project',str(out),'--name',key,'--exist-ok','--save-txt','--save-conf']
    raise ValueError('Unknown '+str(key))


def validate_data_yaml(key,path,model_path=None):
    import yaml
    from pathlib import Path
    from external_paper_baselines import parse_names
    data=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,dict):raise ValueError('Dataset YAML is not mapping')
    if key=='superyolo_hbb':
        for item in ('train','val','nc','names'):
            if item not in data:raise ValueError('Missing SuperYOLO '+item)
        for split in ('train','val'):
            p=Path(data[split]);
            if not p.is_file():raise FileNotFoundError(p)
            for name in p.read_text().splitlines():
                p=Path(name+'_co.png')
                if not p.is_file() or not Path(name+'_ir.png').is_file():
                    raise FileNotFoundError('SuperYOLO list entries must be absolute source image stems: '+name)
                if not Path(str(p).replace('/images/','/labels/').replace('_co.png','.txt')).is_file():
                    raise FileNotFoundError('SuperYOLO paired annotation for '+name)
                expected_dir='VEDAI_1024' if split=='train' else 'VEDAI'
                if expected_dir not in p.parts: raise ValueError('SuperYOLO SR data source wrong resolution tier for '+split)
        if Path(path).name != 'SRvedai.yaml': raise ValueError('SuperYOLO author train.py chooses SR paired loader only when dataset filename is SRvedai.yaml')
        manifest=Path(path).parent/'superyolo_manifest.json'
        if not manifest.is_file():raise FileNotFoundError('Missing audited SuperYOLO SR data manifest')
        if len(data['names'])!=int(data['nc']):raise ValueError('Wrong class count')
    elif key in ('icafusion_hbb','cft_hbb'):
        for v in ('path','train_rgb','train_ir','val_rgb','val_ir','nc','names'):
            if v not in data:raise ValueError('Missing paired data '+v)
        if len(data['names'])!=int(data['nc']):raise ValueError('Wrong class count')
        for v in ('train_rgb','train_ir','val_rgb','val_ir'):
            if not Path(data[v]).is_dir():raise FileNotFoundError(data[v])
        if key=='icafusion_hbb':
            folder=Path(data['path'])/'labels/test'
            if not folder.is_dir():raise FileNotFoundError(folder)
            expected={p.stem for p in Path(data['val_rgb']).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')}
            actual={p.stem for p in folder.glob('*.txt')}
            if not expected or expected!=actual:raise ValueError('ICAFusion labels/test must exactly match CURRENT val_rgb/eval split')
        if model_path:
            name=Path(model_path).name.lower(); source=Path(model_path).read_text(encoding='utf-8').lower()
            if key=='cft_hbb' and 'transformerx3' not in name and 'transfomerx3' not in name:
                raise ValueError('CFT paper requires documented transformerx3 config; cannot infer from generic transformer')
            if key=='icafusion_hbb' and not any(t in (name+' '+source) for t in ('dmff','ica','cross','transfusion','fusion')):
                raise ValueError('ICAFusion selected config has no visible multimodal fusion marker')
    elif key=='c2former_obb':
        if not str(path).endswith('.yaml') and not str(path).endswith('.py'):pass
    return data


def prepare_plan(args):
    key=args.method;upstream=(Path(args.workspace).resolve()/key)
    receipt=verify_repo(key,upstream)
    data=Path(args.data).resolve();model=Path(args.model).resolve()
    if not data.is_file() or not model.is_file():raise FileNotFoundError('Missing dataset/model config')
    if upstream not in model.parents: raise ValueError('The model YAML/Python config must come from the original official checkout')
    if args.weights and not Path(args.weights).is_file():raise FileNotFoundError(args.weights)
    if args.phase=='test' and (not args.checkpoint or not Path(args.checkpoint).is_file()):
        raise FileNotFoundError('Real checkpoint required')
    if key=='c2former_obb':
        txt=model.read_text(encoding='utf-8')
        if 'Two_Stream_S2ANet' not in txt and '_base_' not in txt:raise ValueError('Not original C2Former base/child config')
        if 'model = dict(fam_head=dict(num_classes=' not in txt:
            raise ValueError('Must use generated paired/class-adjusted child config from make_c2former_config')
        if not args.ack_dataset_adapter:raise ValueError('--ack-dataset-adapter required after reviewing generated C2Former view')
    elif key=='superyolo_hbb':
        if model.name!='SRyolo_MF.yaml':raise ValueError('Full-paper SuperYOLO needs original SRyolo_MF.yaml')
        validate_data_yaml(key,data,model)
        if not (upstream.parent/(key+'.io_patch.json')).is_file():
            raise ValueError('SuperYOLO I/O source path fix required (official_upstream_compat.py)')
        if args.phase=='train':
            if not args.super_resolution or not args.ack_sr_data:
                raise ValueError('SuperYOLO full method requires --super-resolution and --ack-sr-data')
    else:
        validate_data_yaml(key,data,model)
        if key=='icafusion_hbb' and not (upstream.parent/(key+'.io_patch.json')).is_file():
            raise ValueError('ICAFusion original test path must be patched via official_upstream_compat.py')
        if not args.ack_paper_arch:raise ValueError('Manual upstream architecture inspection required: --ack-paper-arch')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    cmd=command_for(key,python=args.python,data=data,model=model,out=out,
                    epochs=args.epochs,batch=args.batch,imgsz=args.imgsz,
                    weights=args.weights,device=args.device,
                    super_resolution=args.super_resolution,
                    phase=args.phase,checkpoint=args.checkpoint)
    report={**receipt,'phase':args.phase,'status':'planned_not_evaluated','geometry':get_method(key)['prediction_geometry'],
            'model':str(model),'model_sha256':sha256(model),'data':str(data),'data_sha256':sha256(data),
            'command':cmd,'weights':args.weights,'checkpoint':str(args.checkpoint) if args.checkpoint else None,
            'warning':'Original model has NOT been trained or scientifically validated here; common held-out evaluator required.'}
    report_file=out/(key+'.'+args.phase+'.plan.json')
    if args.execute:
        report['status']='running';report_file.write_text(json.dumps(report,indent=2)+'\n')
        log=out/(key+'.'+args.phase+'.log')
        with log.open('w',encoding='utf-8') as f:
            p=subprocess.run(cmd,cwd=upstream,stdout=f,stderr=subprocess.STDOUT,check=False,
                             env={**os.environ,'WANDB_MODE':'offline','PYTHONHASHSEED':'0'})
        report['exit_code']=p.returncode;report['log']=str(log)
        report['status']='upstream_command_finished_NOT_common_evaluated' if p.returncode==0 else 'upstream_failed'
        report_file.write_text(json.dumps(report,indent=2)+'\n')
        if p.returncode:raise RuntimeError('Upstream failed, see '+str(log))
    else:report_file.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=load_registry(),required=True)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--python',default=sys.executable)
    p.add_argument('--phase',choices=('train','test'),default='train');p.add_argument('--checkpoint',type=Path)
    p.add_argument('--weights',default='');p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--imgsz',type=int,default=512)
    p.add_argument('--device',default='0');p.add_argument('--super-resolution',action='store_true')
    p.add_argument('--ack-sr-data',action='store_true');p.add_argument('--ack-dataset-adapter',action='store_true')
    p.add_argument('--ack-paper-arch',action='store_true');p.add_argument('--execute',action='store_true')
    args=p.parse_args();r=prepare_plan(args)
    print(shlex.join(r['command']));print(json.dumps({'status':r['status'],'plan':str(Path(args.out)/(args.method+'.'+args.phase+'.plan.json'))},indent=2))
if __name__=='__main__':main()
