"""Automated official-source C2Former 3-seed pipeline in a LEGACY MMRotate env.

Runs no custom architecture. Requires --python pointing to an MMRotate 0.3.x
interpreter with suitable torch/CUDA/mmcv wheels. Kaggle's current runtime
cannot be assumed compatible. Failure is explicit; never substitute YOLO.
Use the SAME uniform-JPEG heldout data.yaml used for proposal and ETFNet.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,hashlib,re
from pathlib import Path
from external_paper_baselines import fetch,sha256
from official_baseline_datasets import c2former_dota_view,make_c2former_config
from paper_c2former_convert import convert
from paper_benchmark_eval import evaluate
from paper_benchmark_stats import generate


def call(args,cwd=None,log=None):
    args=list(map(str,args));print('$',' '.join(args),flush=True)
    if log:
        log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as f:p=subprocess.run(args,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(f'Exit {p.returncode}; see {log}: '+log.read_text(errors='replace')[-3000:])
    else:subprocess.run(args,cwd=cwd,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',type=Path,required=True,help='Existing isolated legacy MMRotate Python executable')
    p.add_argument('--data',type=Path,required=True,help='Uniform-JPEG heldout data.yaml')
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--repo-sha',default='')
    p.add_argument('--eval-only',action='store_true',help='Reuse existing official checkpoints for corrupted-test evaluation')
    p.add_argument('--condition',default='clean',help='Safe condition ID used to isolate prediction directories')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.condition):raise ValueError('Unsafe condition ID')
    python=a.python.resolve();workspace=a.workspace.resolve();workspace.mkdir(parents=True,exist_ok=True)
    if not python.is_file():raise FileNotFoundError('Legacy Python interpreter missing: '+str(python))
    # Fail in seconds if this environment has not been prepared; don't waste GPU hours.
    call([python,'-c','import torch,mmcv,mmdet,mmrotate;'
          'assert mmrotate.__version__.startswith("0.3."),mmrotate.__version__;'
          'assert torch.cuda.is_available(),"No available CUDA"'])
    upstreams=workspace/'official_sources'
    receipt=fetch('c2former_obb',upstreams)
    if a.repo_sha and receipt['source_commit']!=a.repo_sha:raise RuntimeError('Official C2Former commit SHA mismatch')
    repo=upstreams/'c2former_obb'
    view=(workspace/'corruption_views'/a.condition if a.eval_only else workspace/'c2former_jpeg_view')
    if not (view/'c2former_dota_manifest.json').is_file():c2former_dota_view(a.data,view)
    data=json.loads((view/'c2former_dota_manifest.json').read_text())
    names=data['classes']
    # The child config changes only classes / paths, preserves original paper model.
    child=(repo/'configs/s2anet'/f'sjpa_c2former_{a.condition}.py' if a.eval_only else repo/'configs/s2anet/sjpa_c2former_paired_local.py')
    if not child.is_file():make_c2former_config(repo,view,names,output=child)
    test_ids=view/'source_id_map.json'
    results=workspace/'paper_c2former_results';results.mkdir(exist_ok=True)
    all_receipts=[]
    for seed in (0,1,2):
        trained=results/f'seed{seed}'
        out=(trained/'robustness'/a.condition if a.eval_only else trained)
        out.mkdir(parents=True,exist_ok=True)
        # A "latest.pth" file alone does not demonstrate full training.
        # Inspect its MMRotate epoch metadata before reuse and resume if needed.
        ckpt=trained/'latest.pth'
        def completed_epoch(path):
            if not path.is_file():return 0
            snippet=("import torch,sys; c=torch.load(sys.argv[1],map_location='cpu'); "
                     "print(int(c.get('meta',{}).get('epoch',0)))")
            try:
                value=subprocess.check_output([str(python),'-c',snippet,str(path)],
                                              text=True,stderr=subprocess.STDOUT)
                return int(value.strip().splitlines()[-1])
            except Exception as exc:
                raise RuntimeError('Cannot verify C2Former checkpoint epoch: '+str(path)) from exc
        last_epoch=completed_epoch(ckpt)
        if not a.eval_only and last_epoch<a.epochs:
            command=[python,'tools/train.py',child,'--work-dir',trained,'--seed',seed,
                     '--deterministic','--cfg-options',f'runner.max_epochs={a.epochs}']
            if ckpt.is_file():command+=['--resume-from',ckpt]
            call(command,cwd=repo,log=trained/'train.log')
            last_epoch=completed_epoch(ckpt)
        if last_epoch<a.epochs:
            raise RuntimeError(f'C2Former incomplete: seed {seed}, epoch {last_epoch}/{a.epochs}')
        predicted=out/'predictions.pkl'
        if not predicted.is_file():
            call([python,'tools/test.py',child,ckpt,'--out',predicted],cwd=repo,log=out/'test.log')
        json_pred=out/'predictions.json'
        convert(predicted,test_ids,json_pred,num_classes=len(names))
        receipt_path=out/'common_obb_receipt.json'
        r=evaluate(a.data,json_pred,ckpt,'c2former_obb',seed,
                   receipt['source_commit'],receipt_path)
        # Images are the same uniform JPEG bytes, only file naming changes.
        r['input_protocol_note']='C2Former source map remaps JPEG input IDs; output mapped back to common test stems'
        receipt_path.write_text(json.dumps(r,indent=2)+'\n')
        all_receipts.append(r)
    receipts_file=(results/f'robustness_{a.condition}_receipts.json' if a.eval_only else results/'receipts.json')
    receipts_file.write_text(json.dumps(all_receipts,indent=2)+'\n')
    print(f'C2Former {a.condition}: measured 3 seeds from official source. Merge receipts only after contract validation.')

if __name__=='__main__':main()
