"""Kaggle entry: upload the FULL corrected ZIP as dataset, enable Internet/GPU.

The one-click reproducible VEDAI cross-dataset benchmark compares:
  - Proposed MR-SJPA (local corrected source)
  - ETFNet (original authors' GitHub implementation)
Both use three paired seeds, the same JPEG normalized RGB/NIR test pairs,
shared polygon-IoU AP101, and statistical reporting.

C2Former is a separate optional legacy-MMRotate benchmark (see README). It is
NOT silently omitted from a claimed 3-method result: this Kaggle job covers
only the 2 named models and says so in its provenance.
"""
from pathlib import Path
import os,sys,zipfile,subprocess

OUTPUT=Path('/kaggle/working')
INPUTS=Path('/kaggle/input')
EPOCHS=int(os.getenv('ETFNET_EPOCHS','100'))
BATCH=int(os.getenv('ETFNET_BATCH','4'))
AUTHOR_SHA=os.getenv('ETFNET_AUTHOR_SHA','')

matches=[]
for archive in INPUTS.rglob('*.zip'):
    try:
        with zipfile.ZipFile(archive) as z:
            if any(n.endswith('/paper_benchmark_runner.py') for n in z.namelist()):
                matches.append(archive)
    except (OSError,zipfile.BadZipFile):pass
if len(matches)!=1:
    raise RuntimeError(f'Upload EXACTLY ONE corrected full repository ZIP. Candidates: {matches}')

# The benchmark orchestrator lives inside the uploaded FULL ZIP; use trusted
# project code rather than an unrelated prior Kaggle version or GitHub main.
full=matches[0]
root=OUTPUT/'paper_repo'
if not (root/'paper_benchmark_runner.py').is_file():
    root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(full) as z:
        locations=[x for x in z.namelist() if x.endswith('/paper_benchmark_runner.py')]
        if len(locations)!=1:raise RuntimeError('Ambiguous source root')
        prefix=locations[0][:-len('paper_benchmark_runner.py')]
        for entry in z.infolist():
            if not entry.filename.startswith(prefix) or entry.is_dir():continue
            rel=entry.filename[len(prefix):]
            if not rel or '..' in Path(rel).parts:raise ValueError('Unsafe input ZIP member')
            target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(entry) as reader,target.open('wb') as writer:
                import shutil
                shutil.copyfileobj(reader,writer)
cmd=[sys.executable,root/'paper_benchmark_runner.py','--repo',root,
     '--work',OUTPUT,'--epochs',EPOCHS,'--batch',BATCH,'--imgsz','512',
     '--device','0','--accept-license']
if AUTHOR_SHA:cmd.extend(['--etfnet-sha',AUTHOR_SHA])
if os.getenv('C2FORMER_PYTHON'):
    cmd.extend(['--c2former-python',os.environ['C2FORMER_PYTHON']])
subprocess.run(list(map(str,cmd)),cwd=root,check=True)
print('Finished result ZIP:',OUTPUT/'PAPER_ONLY_THREE_SEED_RESULTS.zip')
