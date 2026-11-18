"""Kaggle one-cell entry to the 111-job paper research queue.

Upload Ty_Linh_111_Experiments_Full_Corrected.zip as a Kaggle Dataset.
Enable Internet and GPU. Keep the Kaggle output as a Dataset between sessions
and set PREVIOUS_RESEARCH_DATASET_DIR below before resuming from a new session.

The default full plan requires a preinstalled compatible MMRotate 0.3.x Python
interpreter for authors' C2Former. Set EXCLUDE_C2FORMER=1 to explicitly reduce
to 102 runs; this is not silently replaced by a fake baseline.
"""
from __future__ import annotations
import os,shutil,subprocess,sys,zipfile
from pathlib import Path

# Experiment controls. Every subsequent independent Kaggle session may reuse
# the same values; do not change split seed, dataset hashes or training budget.
STAGE=os.getenv('RESEARCH_STAGE','all')
DATASETS=os.getenv('RESEARCH_DATASETS','vedai dronevehicle m3fd').split()
MAX_NEW_TRAINING_JOBS=int(os.getenv('MAX_NEW_TRAINING_JOBS','8'))
MAX_NEW_EVALUATIONS=int(os.getenv('MAX_NEW_EVALUATIONS','63'))
EXCLUDE_C2FORMER=os.getenv('EXCLUDE_C2FORMER','0')=='1'
C2FORMER_PYTHON=os.getenv('C2FORMER_PYTHON','')
ETFNET_SHA=os.getenv('ETFNET_AUTHOR_SHA','')
PREVIOUS_RESEARCH_DATASET_DIR=os.getenv('PREVIOUS_RESEARCH_DATASET_DIR','')
WORK=Path('/kaggle/working')
REPO=WORK/'Ty_Linh_Paper_Research'
ZIP_PATTERN='Ty_Linh_111_Experiments_Full_Corrected.zip'


def run(args,cwd=None):
    print('$',' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=cwd,check=True)


def extract_exact(archive,destination):
    if (destination/'layout_manager.py').exists():return
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        roots={n.split('/')[0] for n in names if n.endswith('/layout_manager.py')}
        if len(roots)!=1:raise RuntimeError('Expected exactly one source root in corrected ZIP')
        root=next(iter(roots))+'/';destination.mkdir(parents=True,exist_ok=True)
        for member in z.infolist():
            if not member.filename.startswith(root) or member.is_dir():continue
            rel=Path(member.filename[len(root):])
            if '..' in rel.parts or rel.is_absolute():raise ValueError('Unsafe ZIP member')
            out=destination/rel;out.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member) as src,out.open('wb') as dst:shutil.copyfileobj(src,dst)


def main():
    candidates=list(Path('/kaggle/input').rglob(ZIP_PATTERN))
    if len(candidates)!=1:raise FileNotFoundError(f'Upload one Kaggle Dataset containing {ZIP_PATTERN}; got {len(candidates)}')
    extract_exact(candidates[0],REPO)
    run([sys.executable,'layout_manager.py','check'],cwd=REPO)
    run([sys.executable,'layout_manager.py','restore'],cwd=REPO)
    run([sys.executable,'-m','pip','install','-q','-r',REPO/'requirements-data.txt'])
    run([sys.executable,'-m','pip','install','-q','--no-deps','-e',REPO])
    # This is NOT background persistence: Kaggle output datasets are explicit.
    if PREVIOUS_RESEARCH_DATASET_DIR:
        previous=Path(PREVIOUS_RESEARCH_DATASET_DIR)/'research'
        if not previous.is_dir():raise FileNotFoundError('No previous research folder: '+str(previous))
        target=WORK/'research'
        if target.exists():raise FileExistsError('Will not overwrite current training evidence; remove duplicate input')
        shutil.copytree(previous,target)
        previous_datasets=Path(PREVIOUS_RESEARCH_DATASET_DIR)/'datasets'
        if previous_datasets.is_dir():
            if (WORK/'datasets').exists():raise FileExistsError('Cannot overwrite current dataset cache')
            shutil.copytree(previous_datasets,WORK/'datasets')
    cmd=[sys.executable,REPO/'paper_research_pipeline.py','--repo',REPO,'--work',WORK,
         '--stage',STAGE,'--datasets',*DATASETS,'--accept-license',
         '--max-new-training-jobs',MAX_NEW_TRAINING_JOBS,
         '--max-new-evaluations',MAX_NEW_EVALUATIONS]
    if EXCLUDE_C2FORMER:cmd.append('--without-c2former')
    elif C2FORMER_PYTHON:cmd+=['--c2former-python',C2FORMER_PYTHON]
    else:
        raise RuntimeError('C2Former authors code requires an existing MMRotate 0.3.x environment: '
                           'set C2FORMER_PYTHON or EXCLUDE_C2FORMER=1. No fake substitute will be used.')
    if ETFNET_SHA:cmd+=['--etfnet-sha',ETFNET_SHA]
    run(cmd,cwd=REPO)
    print('Session output: /kaggle/working/research (save as a Kaggle Dataset for later sessions)')

if __name__=='__main__':main()
