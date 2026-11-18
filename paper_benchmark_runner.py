"""Auto-acquire selected published RGB/IR datasets and benchmark the proposal vs authors' ETFNet.

Full successful runtime requires Kaggle GPU, Internet, usable authors' source,
compatible dependencies and license acceptance. Incompatible author code stops
with an explicit error rather than inventing a surrogate paper baseline.
Original ETFNet paper scores are NOT reproduced; these are new per-dataset OBB splits.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path

METHODS={'mr_sjpa_proposal':'ultralytics/cfg/models/etfnet/etfnet_P2_C3k2_SJPA_TR_Corrected.yaml',
         'etfnet_official_paper':'ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_TGF.yaml'}
AUTHOR='https://github.com/lansimtech/ETFNet.git'


def run(cmd,*,cwd=None,env=None,log=None):
    cmd=[str(x) for x in cmd]
    print('$',' '.join(cmd),flush=True)
    if log:
        log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as f:
            result=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            tail=log.read_text(errors='replace')[-5000:]
            raise RuntimeError(f'Command failed ({result.returncode}): {cmd}\n{tail}\nFull log: {log}')
    else:subprocess.run(cmd,cwd=cwd,env=env,check=True)


def get_source_from_uploaded_zip(zip_path,destination):
    if destination.exists() and (destination/'layout_manager.py').exists():return destination
    with zipfile.ZipFile(zip_path) as z:
        paths=[p for p in z.namelist() if p.endswith('layout_manager.py')]
        if len(paths)!=1:raise RuntimeError('ZIP does not contain exactly one expected source root')
        prefix=paths[0][:-len('layout_manager.py')]
        if any(name.startswith('/') or '..' in Path(name).parts for name in z.namelist()):
            raise ValueError('Unsafe ZIP path')
        destination.mkdir(parents=True,exist_ok=True)
        for entry in z.infolist():
            if not entry.filename.startswith(prefix) or entry.is_dir():continue
            rel=entry.filename[len(prefix):]
            if not rel:continue
            target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(entry) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    return destination


def official_source(path,pin=''):
    if not (path/'.git').exists():
        run(['git','clone','--depth','1',AUTHOR,path])
    if pin:
        run(['git','fetch','--depth','1','origin',pin],cwd=path)
        run(['git','checkout','--detach','FETCH_HEAD'],cwd=path)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
    origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=path,text=True).strip()
    if origin.rstrip('/').removesuffix('.git').lower()!=AUTHOR.rstrip('/').removesuffix('.git').lower():
        raise RuntimeError('Not the ETFNet authors official repository: '+origin)
    expected=path/METHODS['etfnet_official_paper']
    if not expected.is_file():
        raise FileNotFoundError('Author source does not provide intended paper architecture: '+str(expected))
    return sha


def check_completed(run_path,epochs):
    best=run_path/'weights/best.pt';csv_file=run_path/'results.csv'
    if not best.is_file() or not csv_file.is_file():return False
    with csv_file.open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows:return False
    last=rows[-1];epoch=next((v for k,v in last.items() if k.strip().lower()=='epoch'),None)
    if epoch is None:return False
    return float(epoch)>=epochs-1


def filehash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()


def pipeline(a):
    work=Path(a.work).resolve();work.mkdir(parents=True,exist_ok=True)
    result=work/'paper_e2e_results'/a.dataset;result.mkdir(parents=True,exist_ok=True)
    repo=Path(a.repo).resolve() if a.repo else work/'MR_SJPA_repository'
    if a.zip:
        get_source_from_uploaded_zip(a.zip,repo)
    if not (repo/'layout_manager.py').is_file():
        raise FileNotFoundError('Pass --repo with restored source or --zip with corrected repo')
    run([sys.executable,'layout_manager.py','restore'],cwd=repo)
    run([sys.executable,'layout_manager.py','check'],cwd=repo)
    # The author's copy is NEVER replaced with our local (uncertified) paper replica.
    upstream=work/'ETFNet_official_author'
    sha=official_source(upstream,a.etfnet_sha)
    if a.require_pinned_source and not a.etfnet_sha:raise ValueError('Set --etfnet-sha to lock original author commit')
    run([sys.executable,'-m','pip','install','--no-deps','-e','.'],cwd=repo)
    # Optional providers required for multi-file official VEDAI download.
    run([sys.executable,'-m','pip','install','-q','gdown>=6,<7','py7zr>=0.22,<2','huggingface_hub>=0.25,<2'])
    data_root=work/'datasets'
    if not a.accept_license:
        raise PermissionError('Review license terms for the selected dataset and pass --accept-license')
    run([sys.executable,repo/'paper_three_datasets.py','--datasets',a.dataset,
         '--output',data_root,'--accept-license','--split-seed',str(a.split_seed),
         '--test-fraction',str(a.test_fraction)],cwd=repo,
         log=result/'logs/data_download_and_preprocess.log')
    split_yaml=data_root/a.dataset/'paper_heldout'/'data.yaml'
    if not split_yaml.is_file():
        raise RuntimeError(f'Dataset {a.dataset} preparation did not produce test-ready data.yaml')
    run([sys.executable,repo/'etfnet_cli.py','check-data','--data',split_yaml,'--task','obb',
         '--fingerprint','sha256','--output',result/'dataset_audit.json'],cwd=repo)
    if not json.loads((result/'dataset_audit.json').read_text())['ok']:
        raise RuntimeError('Data validation failed')
    methods={'mr_sjpa_proposal':repo,'etfnet_official_paper':upstream}
    provenance={'source_zip_sha256':filehash(a.zip) if a.zip else None,'proposal_source':str(repo),
                'official_etfnet_url':AUTHOR,'official_etfnet_commit':sha,
                'official_etfnet_model_sha256':filehash(upstream/METHODS['etfnet_official_paper']),
                'dataset':a.dataset, 'dataset_yaml':str(split_yaml),'heldout_manifest_sha256':filehash(split_yaml.parent/'heldout_manifest.json'),
                'training':{'epochs':a.epochs,'batch':a.batch,'imgsz':a.imgsz,'device':a.device},
                'seeds':[0,1,2], 'scope':f'{a.dataset} new paired heldout OBB comparison, not original ETFNet metric reproduction',
                'status':'in_progress'}
    (result/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    worker=work/'paper_source_isolated_worker.py'
    shutil.copy2(repo/'paper_benchmark_worker.py',worker)
    receipts=[]
    for seed in (0,1,2):
        for name,source in methods.items():
            p=result/name/f'seed{seed}';p.mkdir(parents=True,exist_ok=True)
            config=source/METHODS[name]
            train_dir=p/'train'
            env=os.environ.copy();env['PYTHONPATH']=str(source)
            env.update({'WANDB_MODE':'disabled','PYTHONHASHSEED':str(seed),'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
            imported=subprocess.check_output([sys.executable,'-c','import ultralytics;print(ultralytics.__file__)'],
                cwd=source,env=env,text=True).strip()
            if not Path(imported).resolve().is_relative_to(source.resolve()):
                raise RuntimeError('Wrong source imported for '+name+': '+imported)
            if not check_completed(train_dir,a.epochs):
                resume=''
                for filename in ('weights/last-resume.pt','weights/last.pt'):
                    if (train_dir/filename).is_file():resume=str(train_dir/filename);break
                command=[sys.executable,worker,'--phase','train','--source',source,
                    '--model',config,'--data',split_yaml,'--project',p,'--name','train',
                    '--seed',seed,'--epochs',a.epochs,'--batch',a.batch,'--imgsz',a.imgsz,'--device',a.device]
                if resume:command+=['--resume',resume]
                run(command,cwd=source,env=env,log=result/'logs'/f'{name}_seed{seed}_train.log')
            if not check_completed(train_dir,a.epochs):
                raise RuntimeError(f'Incomplete {name} seed{seed} training; refusing to score')
            best=train_dir/'weights/best.pt'
            metrics_path=p/'test_worker.json';pred=p/'test_eval/predictions.json'
            if not metrics_path.is_file() or not pred.is_file():
                run([sys.executable,repo/'paper_benchmark_worker.py', '--phase','test','--source',source,
                    '--model',best,'--data',split_yaml,'--project',p,'--name','test_eval',
                    '--seed',seed,'--epochs',a.epochs,'--batch',a.batch,'--imgsz',a.imgsz,
                    '--device',a.device,'--metrics-path',metrics_path],
                    cwd=source,env=env,log=result/'logs'/f'{name}_seed{seed}_test.log')
            checked=json.loads(metrics_path.read_text())
            if checked['status']!='completed' or checked['split']!='test':raise RuntimeError('Missing independent test evidence')
            out=p/'common_obb_receipt.json'
            # Re-evaluate on every run from saved predictions. Never trust training CSV's best val epoch.
            run([sys.executable,repo/'paper_benchmark_eval.py','--data',split_yaml,'--pred',pred,
                 '--checkpoint',best,'--method',name,'--seed',seed,
                 '--source-commit',sha if name=='etfnet_official_paper' else filehash(repo/'pyproject.toml'),
                 '--out',out], cwd=repo,log=result/'logs'/f'{name}_seed{seed}_common_eval.log')
            receipts.append(json.loads(out.read_text()))
            (result/'completed_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    sys.path.insert(0,str(repo))
    from paper_benchmark_stats import generate
    stat=generate(receipts,result/'paper_statistics',dataset_name=a.dataset)
    provenance['c2former_status']='not_executed_no_compatible_legacy_environment'
    if a.c2former_python:
        run([sys.executable,repo/'paper_c2former_3seed.py','--python',a.c2former_python,
             '--data',split_yaml,'--workspace',work/f'c2former_workspace_{a.dataset}','--epochs',a.epochs],
            cwd=repo,log=result/'logs/c2former_three_seed.log')
        c2_receipts=json.loads((work/f'c2former_workspace_{a.dataset}/paper_c2former_results/receipts.json').read_text())
        receipts.extend(c2_receipts)
        stat=generate(receipts,result/'paper_statistics',dataset_name=a.dataset)
        provenance['c2former_status']='completed_with_common_OBB_evaluator'
    provenance['status']='complete';provenance['statistical_report']=str(result/'paper_statistics/statistical_report.json')
    (result/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    shutil.make_archive(str(work/f'PAPER_ONLY_{a.dataset.upper()}_THREE_SEED_RESULTS'),'zip',result)
    print('Completed: '+str(work/f'PAPER_ONLY_{a.dataset.upper()}_THREE_SEED_RESULTS.zip'))
    return stat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',type=Path,help='Full corrected uploaded ZIP')
    p.add_argument('--dataset',choices=('vedai','dronevehicle','m3fd'),default='vedai')
    p.add_argument('--all-datasets',action='store_true',help='Run three paper datasets sequentially; needs substantial storage')
    p.add_argument('--repo',type=Path,help='Already extracted corrected project')
    p.add_argument('--work',type=Path,default=Path('/kaggle/working'))
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--imgsz',type=int,default=512)
    p.add_argument('--device',default='0')
    p.add_argument('--etfnet-sha',default=os.environ.get('ETFNET_AUTHOR_SHA',''))
    p.add_argument('--c2former-python',type=Path,default=Path(os.environ['C2FORMER_PYTHON']) if os.environ.get('C2FORMER_PYTHON') else None,
                   help='Optional pre-installed legacy MMRotate Python for automatic C2Former 3-seed stage')
    p.add_argument('--require-pinned-source',action='store_true')
    p.add_argument('--accept-license',action='store_true')
    p.add_argument('--split-seed',type=int,default=991)
    p.add_argument('--test-fraction',type=float,default=.2)
    args=p.parse_args()
    if args.epochs<1 or args.batch<1 or args.imgsz<32:raise ValueError('Bad training options')
    for dataset in ('vedai','dronevehicle','m3fd') if args.all_datasets else (args.dataset,):
        args.dataset=dataset
        pipeline(args)

if __name__=='__main__':main()
