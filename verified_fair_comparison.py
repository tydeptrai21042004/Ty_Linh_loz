#!/usr/bin/env python3
"""Fail-closed evaluator receipt checker; never invent metrics.

This tool does NOT compute mAP. Input must be outputs of a COMMON evaluator,
not arbitrary upstream printed numbers. It verifies evidence files and protocol
identity. A passing result is only 'internally consistent claims', not science
verification. Heldout test may only be evaluated after fixed model selection.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path

KNOWN={'sjpa_tr_proposal','etfnet_tgf_paper','c2former_obb','superyolo_hbb','icafusion_hbb','cft_hbb'}
SHARED=('geometry','dataset_digest','split_digest','class_order','evaluator_id','iou_thresholds',
        'input_resolution','evaluation_split','postprocess_protocol','metric_definition')


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def compare(receipts, output):
    rows=[json.loads(Path(p).read_text(encoding='utf-8')) for p in receipts]
    if len(rows)<2: raise ValueError('At least two receipts required')
    contract=None;methods=set();output_rows=[]
    for r in rows:
        method=r.get('method')
        if method not in KNOWN: raise ValueError('Unknown method '+str(method))
        if method in methods:raise ValueError('Repeated method: '+method)
        methods.add(method)
        if r.get('source_kind')!='measured' or r.get('status')!='completed':
            raise ValueError('Only COMPLETED measured runs allowed: '+method)
        for k in SHARED:
            if k not in r:raise ValueError('Missing contract key '+k+' for '+method)
        if r['geometry'] not in ('obb','hbb'):raise ValueError('Geometry must be obb or hbb')
        if r['evaluation_split']!='test':raise ValueError('Not a held-out test: '+method)
        if not isinstance(r['class_order'],list) or not r['class_order']:raise ValueError('Class order missing')
        if len(r['iou_thresholds'])<1:raise ValueError('Missing IoU thresholds')
        c=tuple(json.dumps(r[k],sort_keys=True) for k in SHARED)
        if contract is not None and c!=contract:raise ValueError('UNFAIR: differing geometry/dataset/classes/metrics or postprocessing')
        contract=c
        for k in ('checkpoint','predictions','ground_truth'):
            if not r.get(k) or not Path(r[k]).is_file(): raise FileNotFoundError(f'{method}: missing {k}')
            if r.get(k+'_sha256') != sha256(r[k]): raise ValueError(f'{method}: incorrect sha256 for {k}')
        for k in ('map50','map50_95'):
            v=r.get(k)
            if not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1:
                raise ValueError(f'{method}: invalid {k}')
        output_rows.append({'method':method,'geometry':r['geometry'],'map50':r['map50'],
                            'map50_95':r['map50_95'],'evaluator':r['evaluator_id'],
                            'dataset_digest':r['dataset_digest'],'split_digest':r['split_digest']})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as fh:
        wr=csv.DictWriter(fh,fieldnames=list(output_rows[0]));wr.writeheader();wr.writerows(output_rows)
    return {'status':'receipts_consistent_NOT_externally_verified','methods':sorted(methods),'csv':str(output)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipts',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    print(json.dumps(compare(args.receipts,args.out),indent=2))
if __name__=='__main__':main()
