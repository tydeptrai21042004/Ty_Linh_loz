"""Local structural audit; paper baselines must use AUTHORS' source checkout."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
import yaml
from ultralytics.nn.tasks import OBBModel

PROPOSAL = ROOT / 'ultralytics/cfg/models/etfnet/etfnet_P2_C3k2_SJPA_TR_Corrected.yaml'
PUBLISHED = {
 'ETFNet': {'doi':'10.1016/j.inffus.2026.104658', 'repo':'https://github.com/lansimtech/ETFNet'},
 'C2Former': {'doi':'10.1109/TGRS.2024.3376819', 'repo':'https://github.com/yuanmaoxun/C2Former'},
}

def audit(*,forward=True,imgsz=64):
    cfg=yaml.safe_load(PROPOSAL.read_text())
    layers=[x[2] for x in cfg['backbone']+cfg['head']]
    assert 'MR_SJPA' in layers and cfg['head'][-1][2]=='OBB' and cfg['ch']==6
    out={'proposal':'MR_SJPA','proposal_config':str(PROPOSAL.relative_to(ROOT)),
         'published_paper_baselines':PUBLISHED,'local_dummy_baselines':[],
         'forward_executed':forward,'ok':True,
         'note':'Paper implementation integrity is checked on independent author-source checkouts; no paper metric is implied.'}
    if forward:
        model=OBBModel(cfg,verbose=False).eval()
        with torch.no_grad():
            y=model(torch.zeros(1,6,imgsz,imgsz))
        assert y is not None
        out['proposal_parameters']=sum(p.numel() for p in model.parameters())
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--imgsz',type=int,default=64)
    p.add_argument('--no-forward',action='store_true')
    p.add_argument('--output')
    a=p.parse_args()
    data=audit(forward=not a.no_forward,imgsz=a.imgsz)
    result=json.dumps(data,indent=2)
    print(result)
    if a.output: Path(a.output).write_text(result+'\n')
