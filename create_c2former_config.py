#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from official_baseline_datasets import make_c2former_config
from external_paper_baselines import sha256
p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--view',type=Path,required=True)
p.add_argument('--output',type=Path)
a=p.parse_args()
names=json.loads((a.view/'c2former_dota_manifest.json').read_text())['classes']
config=make_c2former_config(a.repo,a.view,names,output=a.output)
print(json.dumps({'status':'configured_not_trained','config':str(config),'sha256':sha256(config)},indent=2))
