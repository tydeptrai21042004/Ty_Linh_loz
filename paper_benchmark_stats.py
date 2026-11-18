"""Three matched-seed statistical evidence; no hard-coded experimental metrics."""
from __future__ import annotations
import argparse,csv,itertools,json,math
from collections import defaultdict
from pathlib import Path
from statistics import mean,stdev

METRICS=('map50','map50_95')


def _stats(values):
    from scipy.stats import t
    values=list(map(float,values));n=len(values)
    if n<2:raise ValueError('At least two seeds required')
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite measurement')
    sd=stdev(values);radius=t.ppf(0.975,n-1)*sd/math.sqrt(n)
    return {'n':n,'mean':mean(values),'sd':sd,'ci95':[mean(values)-radius,mean(values)+radius]}


def _paired(a,b):
    from scipy.stats import t as student_t
    changes=[x-y for x,y in zip(a,b)]
    summary=_stats(changes);n=len(changes)
    tstat=math.inf if summary['sd']==0 and summary['mean'] else (0. if summary['sd']==0 else summary['mean']/(summary['sd']/math.sqrt(n)))
    p_t=None if summary['sd']==0 else float(2*student_t.sf(abs(tstat),n-1))
    observed=abs(mean(changes))
    total=0;extreme=0
    for signs in itertools.product((-1,1),repeat=n):
        total+=1
        if abs(mean(d*s for d,s in zip(changes,signs)))+1e-14>=observed:extreme+=1
    return {**summary,'paired_differences':changes,'paired_t_p_two_sided':p_t,
            'sign_flip_exact_p_two_sided':extreme/total,
            'sign_flip_resolutions':total,
            'warning':'Three seeds give low-power significance evidence; exact sign-flip p cannot be below 0.25 for n=3.'}


def generate(rows,output,*,proposal='mr_sjpa_proposal',baseline='etfnet_official_paper',required_seeds=(0,1,2)):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    index={};contracts=[]
    for r in rows:
        k=(r['method'],int(r['seed']))
        if k in index:raise ValueError('Duplicate experiment: '+str(k))
        index[k]=r
        if r.get('status')!='completed':raise ValueError('Incomplete measurement '+str(k))
        fields=('evaluator_id','dataset_digest','split_digest','class_order','iou_thresholds','eval_geometry','eval_split')
        if any(f not in r for f in fields):raise ValueError('Missing common evaluator contract '+str(k))
        contracts.append(tuple(json.dumps(r[f],sort_keys=True) for f in fields))
    if len(set(contracts))!=1:raise ValueError('Unmatched dataset/split/evaluator: cannot generate fairness table')
    if rows[0]['eval_split']!='test' or rows[0]['eval_geometry']!='obb':
        raise ValueError('Only common OBB held-out test scores allowed')
    methods={r['method'] for r in rows}
    allowed={proposal,baseline,'c2former_obb'}
    if not {proposal,baseline}.issubset(methods) or not methods.issubset(allowed):
        raise ValueError('Only proposed method and original-paper ETFNet/C2Former baselines permitted')
    ordered=[proposal,baseline]+(['c2former_obb'] if 'c2former_obb' in methods else [])
    seeds=list(map(int,required_seeds))
    if len(seeds)!=3 or len(set(seeds))!=3:raise ValueError('Require exactly 3 unique paired seeds')
    for method in methods:
        if {s for m,s in index if m==method}!=set(seeds):raise ValueError('Missing or unexpected seed for '+method)
    stats={m:{method:_stats([float(index[(method,seed)][m]) for seed in seeds]) for method in ordered} for m in METRICS}
    paired={f'{proposal}_minus_{other}':{m:_paired([float(index[(proposal,seed)][m]) for seed in seeds],
                      [float(index[(other,seed)][m]) for seed in seeds]) for m in METRICS}
            for other in ordered if other!=proposal}
    payload={'scope':'new held-out VEDAI benchmark; not original-paper metric reproduction',
             'seeds':seeds,'methods':ordered,'per_method':stats,'paired':paired,
             'test_evaluated_after_checkpoint_selection':True,
             'protocol':{f:rows[0][f] for f in ('evaluator_id','dataset_digest','split_digest','class_order','iou_thresholds','eval_geometry','eval_split')},
             'p_value_note':'Exploratory paired t and exact sign-flip tests. No significance claim based solely on three seeds.'}
    (output/'statistical_report.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    with (output/'per_seed_test_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','seed',*METRICS,'checkpoint_sha256','source_commit'])
        writer.writeheader()
        for seed in seeds:
            for method in ordered:
                r=index[method,seed];writer.writerow({k:r.get(k,'') for k in writer.fieldnames})
    with (output/'paper_table.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Method','mAP50 mean','mAP50 SD','mAP50-95 mean','mAP50-95 SD','n'])
        for method in ordered:
            w.writerow([method,stats['map50'][method]['mean'],stats['map50'][method]['sd'],
                        stats['map50_95'][method]['mean'],stats['map50_95'][method]['sd'],3])
    lines=['\\begin{table}[t]','\\centering',
           '\\caption{Common held-out OBB evaluation on the new VEDAI split (three paired seeds).}',
           '\\begin{tabular}{lcc}','\\toprule','Method & mAP@0.50 & mAP@0.50:0.95 \\\\', '\\midrule']
    for method in (proposal,baseline):
        f=lambda m:f"{100*stats[m][method]['mean']:.2f} $\\pm$ {100*stats[m][method]['sd']:.2f}"
        lines.append(f'{method.replace("_","\\_")} & {f("map50")} & {f("map50_95")} \\\\')
    lines+=['\\bottomrule','\\end{tabular}',
            '\\label{tab:paper_baselines}', '\\end{table}']
    (output/'paper_table.tex').write_text('\n'.join(lines)+'\n')
    return payload

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipts',required=True,nargs='+');p.add_argument('--out',required=True)
    a=p.parse_args();generate([json.loads(Path(x).read_text()) for x in a.receipts],a.out)
