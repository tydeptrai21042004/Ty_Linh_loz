"""Reproducible MR-SJPA research queue: 3 DOI datasets, 37 training jobs each.

All paper BASELINES are authors' official methods; the five MR-SJPA variations are
component ABLATIONS, not independent published baselines. Training jobs are
restartable across Kaggle sessions using checkpoints and result receipts.

No metrics or training results are fabricated. Incompatible external code,
missing datasets, missing test predictions, or unavailable C2Former environments
raise explicit errors and do not produce publication tables.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

DATASETS = ('vedai', 'dronevehicle', 'm3fd')
SEEDS = (0, 1, 2)
PAPER_MODELS = ('mr_sjpa_proposal', 'etfnet_official_paper', 'c2former_obb')
ABLATIONS = {
    'no_spatial': 1,
    'no_procrustes_spatial_only': 2,
    'no_reliability_gate': 3,
    'no_trust_projection': 4,
    'ir_residual_only': 5,
}
PROPOSAL_MODEL = 'ultralytics/cfg/models/etfnet/etfnet_P2_C3k2_SJPA_TR_Corrected.yaml'
PAPER_MODEL = 'ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_TGF.yaml'
DEFAULTS = {'trust_radius': 0.25, 'coherence_tau': 0.30, 'groups': 32,
            'stat_grid': 16, 'shift_penalty': 0.01, 'shift_margin': 0.01}
VALUES = {'trust_radius': (0.10, 0.25, 0.50),
          'coherence_tau': (0.20, 0.30, 0.40),
          'groups': (16, 32, 64),
          'stat_grid': (8, 16, 24),
          'shift_penalty': (0.00, 0.01, 0.03),
          'shift_margin': (0.00, 0.01, 0.03)}

@dataclass(frozen=True)
class Job:
    id: str
    dataset: str
    stage: str
    method: str
    seed: int
    epochs: int
    change: str = ''
    value: float | int | None = None
    ablation: int = 0


def make_plan(datasets=DATASETS, *, include_c2former=True, sweep_epochs=30, full_epochs=100):
    if sweep_epochs < 1 or full_epochs < 1:raise ValueError('Epoch budgets must be positive')
    if not datasets or len(datasets)!=len(set(datasets)) or any(x not in DATASETS for x in datasets):
        raise ValueError('Datasets must be unique supported names')
    result=[]
    for ds in datasets:
        result.append(Job(f'{ds}__sweep__default',ds,'sweep','mr_sjpa_proposal',0,sweep_epochs))
        for k, vals in VALUES.items():
            for v in vals:
                if v == DEFAULTS[k]:continue
                token=str(v).replace('.','p')
                result.append(Job(f'{ds}__sweep__{k}_{token}',ds,'sweep','mr_sjpa_proposal',0,sweep_epochs,k,v))
        for method in PAPER_MODELS:
            if method == 'c2former_obb' and not include_c2former:continue
            for seed in SEEDS:
                result.append(Job(f'{ds}__final__{method}__s{seed}',ds,'final',method,seed,full_epochs))
        for name,idx in ABLATIONS.items():
            for seed in SEEDS:
                result.append(Job(f'{ds}__ablation__{name}__s{seed}',ds,'ablation',f'ablation_{name}',seed,full_epochs,ablation=idx))
    return result


def write_plan(jobs,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data={'protocol':'MR-SJPA-3-Dataset-PaperOnly-v2', 'datasets':sorted({j.dataset for j in jobs}),
          'seeds':list(SEEDS),'models':[m for m in PAPER_MODELS if any(j.method==m for j in jobs)],
          'jobs':[asdict(j) for j in jobs], 'counts':{
              'total_training':len(jobs), 'sweep':sum(j.stage=='sweep' for j in jobs),
              'final':sum(j.stage=='final' for j in jobs),
              'ablation':sum(j.stage=='ablation' for j in jobs)},
          'scope':'Published author-source baselines; component variants only in ablation table',
          'note':'Training jobs only; clean and corruption evaluations reuse checkpoints.'}
    path.write_text(json.dumps(data,indent=2)+'\n')
    return data


def checksum(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def execute(cmd,*,cwd=None,env=None,log=None):
    cmd=list(map(str,cmd));print('$',' '.join(cmd),flush=True)
    if log:
        log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as fd:
            r=subprocess.run(cmd,cwd=cwd,env=env,stdout=fd,stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError('Command failed: '+' '.join(cmd)+'\n'+log.read_text(errors='replace')[-3500:])
    else:subprocess.run(cmd,cwd=cwd,env=env,check=True)


def load_validation_best(results_csv):
    """Only the VALIDATION mAP from the training log is used in sweep selection."""
    with Path(results_csv).open(newline='') as f: rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No training metrics in '+str(results_csv))
    keys=list(rows[0]); valid=[k for k in keys if 'map50-95' in k.lower().replace(' ','') and ('metrics' in k.lower() or 'val' in k.lower())]
    if len(valid)!=1:raise ValueError('Missing/ambiguous validation mAP50-95 in '+str(keys))
    vals=[float(r[valid[0]]) for r in rows if r.get(valid[0], '').strip()]
    if not vals or not all(0<=x<=1 for x in vals):raise ValueError('Bad validation metrics')
    return max(vals)


def cfg_values(job, winner=None):
    d=dict(DEFAULTS if winner is None else winner)
    if job.stage=='sweep' and job.change:d[job.change]=job.value
    if job.ablation not in range(6):raise ValueError('Invalid ablation')
    return d


def model_config(template, output, names, values, ablation=0):
    import yaml
    output=Path(output)
    original=yaml.safe_load(Path(template).read_text())
    if not isinstance(original,dict) or not original.get('backbone'):raise ValueError('Invalid model YAML')
    matches=[l for l in original['backbone'] if isinstance(l,list) and len(l)>=4 and l[2]=='MR_SJPA']
    if len(matches)!=1:raise ValueError('Expected exactly one proposed MR_SJPA layer')
    l=matches[0]
    # Parsed layer prepends feature channels internally. The original YAML has
    # precisely these nine user args; the tenth is the new ablation ID.
    l[3]=[int(values['groups']),int(values['stat_grid']),0.001,
          float(values['coherence_tau']),10.0,float(values['trust_radius']),
          1,float(values['shift_penalty']),float(values['shift_margin']),int(ablation)]
    original['nc']=len(names)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(yaml.safe_dump(original,sort_keys=False))
    return output


def dataset_names(data_yaml):
    import yaml
    x=yaml.safe_load(Path(data_yaml).read_text())['names']
    if isinstance(x,dict):x=[x[k] for k in sorted(x,key=lambda k:int(k))]
    if not x or len(set(x))!=len(x):raise ValueError('Invalid class order')
    return x


def prepare_dataset(repo,work,dataset,accept_license,split_seed,test_fraction):
    if not accept_license:raise PermissionError('Review dataset license and pass --accept-license')
    from paper_three_datasets import run
    root=work/'datasets'
    run([dataset],output=root,accept_license=True,split_seed=split_seed,test_fraction=test_fraction,
        validation_fingerprint='sha256')
    yamlpath=root/dataset/'paper_heldout/data.yaml'
    if not yamlpath.is_file():raise RuntimeError('No held-out dataset created')
    return yamlpath


def _source_path(repo,work,method,etfnet_sha):
    if method.startswith('ablation_') or method=='mr_sjpa_proposal':return repo
    if method=='etfnet_official_paper':
        from paper_benchmark_runner import official_source
        official_source(work/'ETFNet_official_author',etfnet_sha)
        return work/'ETFNet_official_author'
    raise ValueError('C2Former requires the separate authors MMRotate workflow')


def train_and_maybe_test(job,*,repo,work,data_yaml,config,source,seed_results,device,batch,imgsz,validation_only=False,source_commit=''):
    """Run a real training/validation and optionally one held-out test.

    Completeness requires an actual checkpoint, final results.csv epoch, and,
    for tests, actual predictions.json and a common OBB receipt.
    """
    from paper_benchmark_runner import check_completed
    jobdir=seed_results/job.id;jobdir.mkdir(parents=True,exist_ok=True)
    cfg=Path(config).resolve();data_yaml=Path(data_yaml).resolve();source=Path(source).resolve()
    intent={'job':asdict(job),'data_yaml_sha256':checksum(data_yaml),
            'config_sha256':checksum(cfg),'source_path':str(source),'imgsz':imgsz,'batch':batch}
    intent_path=jobdir/'job_intent.json'
    if intent_path.exists() and json.loads(intent_path.read_text())!=intent:
        raise RuntimeError('Conflicting resumed job settings: '+job.id)
    intent_path.write_text(json.dumps(intent,indent=2)+'\n')
    env=os.environ.copy();env['PYTHONPATH']=str(source)
    env.update(WANDB_MODE='disabled',PYTHONHASHSEED=str(job.seed),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    imported=subprocess.check_output([sys.executable,'-c','import ultralytics;print(ultralytics.__file__)'],cwd=source,env=env,text=True).strip()
    if not Path(imported).resolve().is_relative_to(source):raise RuntimeError('Wrong official source: '+imported)
    worker=repo/'paper_benchmark_worker.py'
    train=jobdir/'train'
    if not check_completed(train,job.epochs):
        resume=next((train/x for x in ('weights/last-resume.pt','weights/last.pt') if (train/x).exists()),None)
        cmd=[sys.executable,worker,'--phase','train','--source',source,'--model',cfg,'--data',data_yaml,
             '--project',jobdir,'--name','train','--seed',job.seed,'--epochs',job.epochs,
             '--batch',batch,'--imgsz',imgsz,'--device',device]
        if resume:cmd+=['--resume',resume]
        execute(cmd,cwd=source,env=env,log=jobdir/'train.log')
    if not check_completed(train,job.epochs):raise RuntimeError('Incomplete training: '+job.id)
    best=train/'weights/best.pt'
    if not best.is_file():raise RuntimeError('Missing checkpoint')
    output={'id':job.id,'method':job.method,'dataset':job.dataset,'seed':job.seed,
            'best_checkpoint':str(best),'checkpoint_sha256':checksum(best),
            'val_best_map50_95':load_validation_best(train/'results.csv'),
            'status':'trained'}
    (jobdir/'training_receipt.json').write_text(json.dumps(output,indent=2)+'\n')
    if validation_only:return output
    return test_trained(job,repo,jobdir,source,data_yaml,best,batch,imgsz,device,source_commit)


def test_trained(job,repo,jobdir,source,data_yaml,best,batch,imgsz,device,source_commit):
    env=os.environ.copy();env['PYTHONPATH']=str(source);env['WANDB_MODE']='disabled'
    pred=jobdir/'test_eval/predictions.json';metrics=jobdir/'test_worker.json';receipt=jobdir/'common_obb_receipt.json'
    if not pred.is_file() or not metrics.is_file():
        execute([sys.executable,repo/'paper_benchmark_worker.py','--phase','test',
                 '--source',source,'--model',best,'--data',data_yaml,'--project',jobdir,'--name','test_eval',
                 '--seed',job.seed,'--epochs',job.epochs,'--batch',batch,'--imgsz',imgsz,
                 '--device',device,'--metrics-path',metrics],
                cwd=source,env=env,log=jobdir/'test.log')
    witness=json.loads(metrics.read_text())
    if witness.get('status')!='completed' or witness.get('split')!='test' or not pred.exists():
        raise RuntimeError('No trustworthy test evidence: '+job.id)
    if witness.get('predictions_sha256') and witness['predictions_sha256']!=checksum(pred):
        raise RuntimeError('Predictions changed since validation')
    execute([sys.executable,repo/'paper_benchmark_eval.py','--data',data_yaml,'--pred',pred,
             '--checkpoint',best,'--method',job.method,'--seed',job.seed,
             '--source-commit',source_commit or 'proposal_local_source','--out',receipt],
            cwd=repo,log=jobdir/'common_eval.log')
    data=json.loads(receipt.read_text())
    if data.get('status')!='completed':raise RuntimeError('Incomplete OBB evaluation')
    return data


def choose_winner(ds, jobs, root):
    rows=[]
    for j in jobs:
        if j.dataset!=ds or j.stage!='sweep':continue
        path=root/j.id/'training_receipt.json'
        if not path.is_file():raise RuntimeError('Complete all thirteen sweeps before selecting hyperparameters: '+j.id)
        r=json.loads(path.read_text())
        if r.get('status')!='trained':raise RuntimeError('Sweep not completed: '+j.id)
        rows.append((float(r['val_best_map50_95']),j))
    if len(rows)!=13:raise RuntimeError('Sweep needs exactly 13 completed trials')
    score,job=sorted(rows,key=lambda t:(-t[0],t[1].id))[0]
    vals=cfg_values(job)
    return {'dataset':ds,'winning_job':job.id,'validation_map50_95':score,
            'parameters':vals,'selection_seed':0,'selection_epochs':job.epochs,
            'evidence':'best validation metric only; test split not used'}


def report_results(ds,jobs,job_root,output,*,include_c2former):
    from paper_benchmark_stats import generate,_stats,_paired
    output.mkdir(parents=True,exist_ok=True)
    selected=[j for j in jobs if j.dataset==ds and j.stage in ('final','ablation')]
    rows={}
    for j in selected:
        receipt=job_root/j.id/'common_obb_receipt.json'
        if not receipt.is_file():raise RuntimeError('Cannot report without actual test predictions: '+j.id)
        rows[j.id]=json.loads(receipt.read_text())
    main=[rows[j.id] for j in selected if j.stage=='final']
    generate(main,output/'paper_comparison',dataset_name=ds)
    from paper_benchmark_stats import _stats
    perclass_rows=[]
    for method in sorted({r['method'] for r in main}):
        group=sorted((r for r in main if r['method']==method),key=lambda r:r['seed'])
        common_classes=set.intersection(*(set(r.get('per_class',{})) for r in group))
        for classname in sorted(common_classes):
            x=_stats([float(r['per_class'][classname]['ap50']) for r in group])
            y=_stats([float(r['per_class'][classname]['map50_95']) for r in group])
            perclass_rows.append([method,classname,x['mean'],x['sd'],y['mean'],y['sd']])
    with (output/'per_class_test_AP.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','class','ap50_mean','ap50_sd','map50_95_mean','map50_95_sd'])
        w.writerows(perclass_rows)
    # Ensure every ablation follows exactly the same held-out evaluation contract.
    keys=('dataset_digest','split_digest','class_order','iou_thresholds','evaluator_id','eval_geometry','eval_split')
    anchor=main[0]
    for r in rows.values():
        if any(r[k]!=anchor[k] for k in keys):raise ValueError('Mismatch between ablation and full reference')
    full={j.seed:rows[j.id] for j in selected if j.method=='mr_sjpa_proposal'}
    table=[];paired={}
    for name in ABLATIONS:
        items=[rows[j.id] for j in selected if j.method=='ablation_'+name]
        items=sorted(items,key=lambda r:r['seed'])
        if len(items)!=3 or set(r['seed'] for r in items)!=set(SEEDS):raise RuntimeError('Missing ablation seed')
        metrics={m:_stats([r[m] for r in items]) for m in ('map50','map50_95')}
        paired[name]={m:_paired([full[s][m] for s in SEEDS],[items[s][m] for s in SEEDS])
                      for m in ('map50','map50_95')}
        table.append({'ablation':name,'map50':metrics['map50'],'map50_95':metrics['map50_95']})
    (output/'ablation_stats.json').write_text(json.dumps({'per_method':table,'paired_vs_full':paired},indent=2)+'\n')
    with (output/'ablation_table.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Ablation','mAP50 mean','mAP50 SD','mAP50-95 mean','mAP50-95 SD'])
        for x in table:w.writerow([x['ablation'],x['map50']['mean'],x['map50']['sd'],x['map50_95']['mean'],x['map50_95']['sd']])
    lines=['\\begin{tabular}{lcc}','\\toprule','MR-SJPA component removal & mAP@0.5 & mAP@0.5:0.95 \\\\','\\midrule']
    for x in table:
        a=x['map50'];b=x['map50_95'];lines.append(f"{x['ablation'].replace('_',' ')} & {100*a['mean']:.2f} $\\pm$ {100*a['sd']:.2f} & {100*b['mean']:.2f} $\\pm$ {100*b['sd']:.2f} \\\\")
    lines+=['\\bottomrule','\\end{tabular}']
    (output/'ablation_table.tex').write_text('\n'.join(lines)+'\n')
    (output/'results_status.json').write_text(json.dumps({'dataset':ds,'completed':True,'n_training_jobs':len([j for j in jobs if j.dataset==ds]),
         'paper_baselines':list(PAPER_MODELS if include_c2former else PAPER_MODELS[:2]),
         'dataset_geometry':'converted_HBB_not_native_OBB' if ds=='m3fd' else 'native_OBB',
         'note':'3 seeds have insufficient exact sign-flip power for p<0.05'},indent=2)+'\n')


def robustness_stage(args,repo,work,ds,jobs,*,budget):
    """Evaluate 20 distortions plus clean on existing final checkpoints, with no retraining."""
    from paper_robustness import conditions,make_view,aggregate
    original=work/'datasets'/ds/'paper_heldout/data.yaml'
    if not original.is_file():raise FileNotFoundError('Run completed dataset preparation first')
    source_sha=checksum(repo/'models/ultra_modeling~nn~modules~block.py')
    base=work/'research'/'runs'/ds
    robustness=work/'research'/'robustness'/ds
    c2_workspace=work/'research'/'c2former'/ds
    all_results=[]
    for condition in conditions():
        name=condition['id']
        view=make_view(original,robustness/'views'/name,condition)
        for method in PAPER_MODELS[:2]:
            source=_source_path(repo,work,method,args.etfnet_sha)
            sha=source_sha if source==repo else subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
            for seed in SEEDS:
                job=next(x for x in jobs if x.dataset==ds and x.stage=='final' and x.method==method and x.seed==seed)
                ckpt=base/job.id/'train/weights/best.pt'
                if not ckpt.is_file():raise RuntimeError('Missing trained final checkpoint '+job.id)
                out=robustness/'predictions'/name/method/f'seed{seed}'
                receipt=out/'common_obb_receipt.json'
                if not receipt.is_file():
                    if budget[0]<=0:return False
                    test_trained(job,repo,out,source,view,ckpt,args.batch,args.imgsz,args.device,sha)
                    budget[0]-=1
                r=json.loads(receipt.read_text())
                all_results.append((name,method,r))
        if not args.without_c2former:
            if not args.c2former_python:raise RuntimeError('Robustness C2Former requires --c2former-python')
            receipt_file=c2_workspace/'paper_c2former_results'/f'robustness_{name}_receipts.json'
            if not receipt_file.is_file():
                if budget[0]<3:return False
                execute([sys.executable,repo/'paper_c2former_3seed.py','--python',args.c2former_python,
                         '--data',view,'--workspace',c2_workspace,'--epochs',args.epochs,
                         '--eval-only','--condition',name],cwd=repo,
                        log=robustness/'logs'/f'c2former_{name}.log')
                budget[0]-=3
            for r in json.loads(receipt_file.read_text()):
                all_results.append((name,'c2former_obb',r))
    aggregate(all_results,work/'research'/'reports'/ds/'robustness')
    return True


def stage_for_dataset(args,repo,work,ds,jobs,budget):
    dsroot=work/'research'/'runs'/ds;dsroot.mkdir(parents=True,exist_ok=True)
    output=work/'research'/'reports'/ds
    data_yaml=prepare_dataset(repo,work,ds,args.accept_license,args.split_seed,args.test_fraction)
    names=dataset_names(data_yaml)
    if args.stage in ('all','sweep'):
        for j in (x for x in jobs if x.dataset==ds and x.stage=='sweep'):
            cfg=model_config(repo/PROPOSAL_MODEL,work/'research'/'configs'/ds/(j.id+'.yaml'),names,cfg_values(j))
            if (dsroot/j.id/'training_receipt.json').is_file():continue
            if budget[0]<=0:return False
            train_and_maybe_test(j,repo=repo,work=work,data_yaml=data_yaml,config=cfg,source=repo,
                 seed_results=dsroot,device=args.device,batch=args.batch,imgsz=args.imgsz,validation_only=True)
            budget[0]-=1
    if args.stage=='sweep':return True
    if args.stage=='robustness':
        return robustness_stage(args,repo,work,ds,jobs,budget=args.evaluation_budget)
    winner=choose_winner(ds,jobs,dsroot)
    output.mkdir(parents=True,exist_ok=True)
    (output/'sweep_winner_validation_only.json').write_text(json.dumps(winner,indent=2)+'\n')
    from paper_research_reports import export_sweep
    export_sweep(ds,jobs,dsroot,output)
    source_sha=checksum(repo/'models/ultra_modeling~nn~modules~block.py')
    for j in (x for x in jobs if x.dataset==ds and x.stage in ('final','ablation')):
        if args.stage=='final' and j.stage!='final':continue
        if args.stage=='ablation' and j.stage!='ablation':continue
        if j.method=='c2former_obb':continue # run as official group below
        source=_source_path(repo,work,j.method,args.etfnet_sha)
        cfg=(model_config(repo/PROPOSAL_MODEL,work/'research'/'configs'/ds/(j.id+'.yaml'),names,
                          winner['parameters'],j.ablation) if source==repo else source/PAPER_MODEL)
        receipt=dsroot/j.id/'common_obb_receipt.json'
        if receipt.is_file():continue
        if budget[0]<=0:return False
        sha=(source_sha if source==repo else subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip())
        train_and_maybe_test(j,repo=repo,work=work,data_yaml=data_yaml,config=cfg,source=source,
             seed_results=dsroot,device=args.device,batch=args.batch,imgsz=args.imgsz,source_commit=sha)
        budget[0]-=1
    if args.stage in ('all','final') and not args.without_c2former:
        # Official MMRotate source, never a YOLO look-alike.
        if not args.c2former_python:
            raise RuntimeError('C2Former needs --c2former-python legacy MMRotate 0.3.x environment. '
                               'Provide it or explicitly choose --without-c2former (then plan = 102 runs).')
        c2dir=work/'research'/'c2former'/ds
        cmd=[sys.executable,repo/'paper_c2former_3seed.py','--python',args.c2former_python,
             '--data',data_yaml,'--workspace',c2dir,'--epochs',args.epochs]
        if not (c2dir/'paper_c2former_results/receipts.json').is_file():
            if budget[0]<3:return False
            execute(cmd,cwd=repo,log=c2dir/'c2former_train.log')
            budget[0]-=3
        c2_receipts=json.loads((c2dir/'paper_c2former_results/receipts.json').read_text())
        for r in c2_receipts:
            j=next(x for x in jobs if x.dataset==ds and x.method=='c2former_obb' and x.seed==r['seed'])
            target=dsroot/j.id/'common_obb_receipt.json';target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(r,indent=2)+'\n')
    if args.stage in ('all','report'):
        report_results(ds,jobs,dsroot,output,include_c2former=not args.without_c2former)
        from paper_research_reports import export_figures
        export_figures(output)
    if args.stage in ('all','robustness'):
        complete=robustness_stage(args,repo,work,ds,jobs,budget=args.evaluation_budget)
        if complete:
            from paper_research_reports import export_figures
            export_figures(output)
        return complete
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--work',type=Path,default=Path('/kaggle/working'))
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--stage',choices=['plan','all','sweep','final','ablation','robustness','report'],default='plan')
    p.add_argument('--without-c2former',action='store_true',help='Explicitly run only two verified-source workflows: 102 training jobs')
    p.add_argument('--c2former-python',type=Path,help='Existing compatible MMRotate 0.3.x CUDA interpreter')
    p.add_argument('--etfnet-sha',default='',help='Pinned authors ETFNet Git commit; strongly recommended')
    p.add_argument('--accept-license',action='store_true')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--sweep-epochs',type=int,default=30)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--imgsz',type=int,default=512)
    p.add_argument('--device',default='0')
    p.add_argument('--split-seed',type=int,default=991)
    p.add_argument('--test-fraction',type=float,default=.2)
    p.add_argument('--max-new-evaluations',type=int,default=999999,
                   help='Bound new condition evaluations per Kaggle session; persist the research folder')
    p.add_argument('--max-new-training-jobs',type=int,default=999999,
                   help='Stop cleanly after this many new jobs; rerun to resume on later Kaggle sessions')
    args=p.parse_args(argv)
    if args.max_new_training_jobs<0 or args.batch<1 or args.imgsz<32:raise ValueError('Invalid options')
    repo=args.repo.resolve();work=args.work.resolve();work.mkdir(parents=True,exist_ok=True)
    jobs=make_plan(args.datasets,include_c2former=not args.without_c2former,
                   sweep_epochs=args.sweep_epochs,full_epochs=args.epochs)
    plan=write_plan(jobs,work/'research'/'experiment_plan.json')
    print('Training plan:',plan['counts'],flush=True)
    if args.stage=='plan':return 0
    if not args.accept_license:raise PermissionError('Pass --accept-license only after checking dataset terms')
    if args.stage in ('all','final') and not args.without_c2former and not args.c2former_python:
        raise ValueError('Missing --c2former-python; use --without-c2former for a deliberately reduced plan')
    execute([sys.executable,repo/'layout_manager.py','restore'],cwd=repo)
    execute([sys.executable,repo/'layout_manager.py','check'],cwd=repo)
    sys.path.insert(0,str(repo))
    budget=[args.max_new_training_jobs]
    args.evaluation_budget=[args.max_new_evaluations]
    for dataset in args.datasets:
        done=stage_for_dataset(args,repo,work,dataset,jobs,budget)
        if not done:
            print('Session budget reached. Preserve /kaggle/working/research and rerun with same settings.',flush=True)
            return 0
    if args.stage in ('all','report'):
        from paper_research_reports import export_cross_dataset
        report_root=work/'research'/'reports'
        export_cross_dataset(report_root,args.datasets,report_root/'combined')
        shutil.make_archive(str(work/'MR_SJPA_PAPER_TABLES_ONLY'),'zip',report_root)
        print('Paper reports archived:',work/'MR_SJPA_PAPER_TABLES_ONLY.zip',flush=True)
    print('Completed requested stages for:',', '.join(args.datasets))
    return 0

if __name__=='__main__':sys.exit(main())
