"""Audit one paper-derived ETFNet baseline, retained ablations and proposal.

This is a structural and numerical forward smoke audit, not a peer-review verification
of external paper implementations or a substitute for full convergence experiments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from ultralytics.nn.tasks import OBBModel, yaml_model_load

MODEL_DIR = ROOT / 'ultralytics/cfg/models/etfnet'
# Description, YAML, channel count, expected fusion module, comparison category.
BASELINES = {
    'etfnet_tgf_paper': ('etfnet_P2_CAFEM_TGF.yaml', 6, 'TGF', 'paper_reproduction'),
    'dual_concat': ('etfnet_dualstream_noCAFEM_noTGF.yaml', 6, 'Concat', 'ablation'),
    'dual_add': ('noCAFEM_noTGF_add.yaml', 6, 'Add', 'ablation'),
    'cafem_only': ('etfnet_CAFEM_noTGF.yaml', 6, 'CAFEM', 'ablation'),
    'tgf_only': ('etfnet_noCAFEM_TGF.yaml', 6, 'TGF', 'ablation'),
    'goci_no_spatial': ('etfnet_P2_CAFEM_GOCI.yaml', 6, 'GOCI', 'ablation'),
    'sjpa_no_reliability': ('etfnet_P2_C3k2_SJPA_TR_Corrected.yaml', 6, 'SJPA', 'ablation'),
    'rtpf_safety_ablation': ('etfnet_P2_C3k2_RTPF.yaml', 6, 'RTPF', 'ablation'),
    'sjpa_tr_proposal': ('etfnet_P2_C3k2_SJPA_TR_Corrected.yaml', 6, 'SJPA', 'proposal'),
}

# Proof of provenance: official authors' open-source model configuration.
PAPER_REPRODUCTION = {
    'etfnet_tgf_paper': {
        'doi': '10.1016/j.inffus.2026.104658',
        'official_repo': 'https://github.com/lansimtech/ETFNet',
        'official_yaml': 'https://raw.githubusercontent.com/lansimtech/ETFNet/main/ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_TGF.yaml',
    }
}


def no_reliability_config(config):
    import copy
    config = copy.deepcopy(config)
    layers = [row for row in config['backbone'] if row[2] == 'SJPA']
    assert len(layers) == 1 and len(layers[0][3]) >= 5
    assert layers[0][3][4] is True, 'The proposed SJPA must enable reliability'
    layers[0][3][4] = False
    return config


def audit(*, forward=True, imgsz=64):
    torch.set_num_threads(min(2, torch.get_num_threads()))
    records = []
    for name, (filename, channels, fusion, category) in BASELINES.items():
        path = MODEL_DIR / filename
        entry = {'baseline': name, 'config': filename, 'channels_expected': channels,
                 'category': category, 'ok': False}
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            config = yaml.safe_load(path.read_text(encoding='utf-8'))
            if name == 'sjpa_no_reliability':
                config = no_reliability_config(config)
            if config['ch'] != channels:
                raise AssertionError(f'channel mismatch: {config["ch"]} != {channels}')
            modules = [row[2] for row in config['backbone'] + config['head']]
            if config['head'][-1][2] != 'OBB':
                raise AssertionError('Not an OBB detection head')
            if fusion is not None and fusion not in modules:
                raise AssertionError(f'missing expected fusion {fusion}')
            if channels == 6 and not any(m in modules for m in ('Concat','Add','SJPA','TGF','GOCI','RTPF','MPCRF')):
                raise AssertionError('Paired baseline has no merge/fusion layer')
            if name in ('dual_concat', 'sjpa_tr_proposal', 'rtpf_safety_ablation', 'sjpa_no_reliability'):
                control = yaml.safe_load((MODEL_DIR / 'etfnet_dualstream_noCAFEM_noTGF.yaml').read_text())
                if config['backbone'][:9] != control['backbone'][:9] or config['backbone'][10:] != control['backbone'][10:] or config['head'] != control['head']:
                    raise AssertionError('Declared matched control differs outside the fusion layer')
                if config['backbone'][9][0:2] != control['backbone'][9][0:2]:
                    raise AssertionError('Fusion connection or repeat count differs')
                if name == 'dual_concat' and config != control:
                    raise AssertionError('Dual-concat is not the reference control')
            entry['model_modules'] = {mod: modules.count(mod) for mod in ('SJPA','RTPF','GOCI','CAFEM','TGF','Add','Concat') if mod in modules}
            if name == 'sjpa_no_reliability':
                entry['reliability_enabled'] = False
            elif fusion == 'SJPA':
                entry['reliability_enabled'] = bool([x for x in config['backbone'] if x[2] == 'SJPA'][0][3][4])
            if forward:
                model = OBBModel(config, verbose=False).eval()
                with torch.inference_mode():
                    output = model(torch.zeros(1, channels, imgsz, imgsz))
                if output is None:
                    raise AssertionError('Forward returned no output')
                entry['parameters'] = sum(p.numel() for p in model.parameters())
                del model, output
            entry['ok'] = True
            if name in PAPER_REPRODUCTION:
                entry['paper_provenance'] = PAPER_REPRODUCTION[name]
        except Exception as err:
            entry['error'] = f'{type(err).__name__}: {err}'
        records.append(entry)
    return {'ok': all(x['ok'] for x in records), 'checked': len(records),
            'forward_executed': forward, 'image_size': imgsz, 'baselines': records,
            'interpretation': 'Published ETFNet YAML manually cross-checked against official author YAML, but this is only a local graph and forward audit. External weight equivalence, full training and published accuracy are not certified.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--forward', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--imgsz', type=int, default=64)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    report=audit(forward=args.forward, imgsz=args.imgsz)
    result=json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + '\n',encoding='utf-8')
    print(result)
    raise SystemExit(0 if report['ok'] else 1)

if __name__ == '__main__':
    main()
