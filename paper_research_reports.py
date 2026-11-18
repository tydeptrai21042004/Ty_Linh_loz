"""Create research tables and PDF figures from measured, completed receipts only."""
from __future__ import annotations
import csv,json
from pathlib import Path


def export_sweep(dataset,jobs,run_root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for job in jobs:
        if job.dataset!=dataset or job.stage!='sweep':continue
        receipt=Path(run_root)/job.id/'training_receipt.json'
        if not receipt.is_file():raise RuntimeError('Missing trained sweep receipt '+job.id)
        r=json.loads(receipt.read_text())
        if r.get('status')!='trained':raise RuntimeError('Incomplete sweep '+job.id)
        rows.append({'trial':job.id,'changed_parameter':job.change or 'default',
                     'value':job.value if job.change else '', 'validation_map50_95':r['val_best_map50_95']})
    if len(rows)!=13:raise ValueError('Require 13 observed screening trials')
    with (out/'sweep_validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['\\begin{tabular}{llr}','\\toprule','Parameter & Value & Validation mAP50:95 \\\\', '\\midrule']
    for r in rows:
        param=r['changed_parameter'].replace('_', ' ')
        val=str(r['value'] if r['value']!='' else 'default')
        lines.append(f"{param} & {val} & {100*float(r['validation_map50_95']):.2f} " + r' \\')
    lines+=['\\bottomrule','\\end{tabular}']
    (out/'sweep_validation.tex').write_text('\n'.join(lines)+'\n')
    return rows


def export_figures(report_dir):
    """Plots are generated only from real report files; missing data is an error."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report_dir=Path(report_dir)
    sweep=list(csv.DictReader((report_dir/'sweep_validation.csv').open()))
    ablations=list(csv.DictReader((report_dir/'ablation_table.csv').open()))
    fig,ax=plt.subplots(figsize=(8,4))
    x=list(range(len(sweep)));y=[100*float(r['validation_map50_95']) for r in sweep]
    ax.plot(x,y,marker='o',linewidth=1)
    ax.set_xticks(x,labels=[r['changed_parameter'].replace('_','\n') for r in sweep],rotation=45,ha='right',fontsize=7)
    ax.set_ylabel('Validation mAP50–95 (%)');ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(report_dir/'sweep_validation.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.barh([r['ablation'].replace('_',' ') for r in ablations],
            [100*float(r['mAP50-95 mean']) for r in ablations],
            xerr=[100*float(r['mAP50-95 SD']) for r in ablations])
    ax.set_xlabel('Held-out OBB mAP50–95 (%)')
    ax.invert_yaxis();ax.grid(alpha=.3,axis='x')
    fig.tight_layout();fig.savefig(report_dir/'ablation_test.pdf',bbox_inches='tight');plt.close(fig)
    robustness=report_dir/'robustness'/'robustness_metrics.csv'
    if robustness.exists():
        data=list(csv.DictReader(robustness.open()))
        fig,ax=plt.subplots(figsize=(8,4))
        methods=sorted({x['method'] for x in data})
        for method in methods:
            points=[r for r in data if r['method']==method and r['condition']!='clean']
            ax.plot(range(len(points)),[-100*float(x['delta_map50_95_from_clean']) for x in points],
                    label=method,marker='.',linewidth=1)
        ax.set_xlabel('Corruption condition index');ax.set_ylabel('mAP50–95 degradation (pp)')
        ax.legend(fontsize=7);ax.grid(alpha=.3)
        fig.tight_layout();fig.savefig(report_dir/'robustness_degradation.pdf',bbox_inches='tight');plt.close(fig)


def export_cross_dataset(results_root,datasets,out):
    """Separate dataset rows: cannot pool class-incompatible datasets as if one test."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for dataset in datasets:
        p=Path(results_root)/dataset/'paper_comparison/statistical_report.json'
        if not p.is_file():raise RuntimeError('No complete final paper statistics for '+dataset)
        obj=json.loads(p.read_text())
        for method in obj['methods']:
            m=obj['per_method']['map50_95'][method]
            rows.append({'dataset':dataset,'method':method,'mean':m['mean'],'sd':m['sd'],
                         'ci95_low':m['ci95'][0],'ci95_high':m['ci95'][1],'n':m['n'],
                         'annotation_geometry':'axis_aligned_HBB_polygon' if dataset=='m3fd' else 'native_OBB'})
    with (out/'three_dataset_paper_table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['\\begin{tabular}{lllcc}','\\toprule','Dataset & Geometry & Paper method & mAP50:95 & 95\\% CI \\\\', '\\midrule']
    for r in rows:
        method=r['method'].replace('_',' ')
        geom='HBB polygon' if r['dataset']=='m3fd' else 'OBB'
        lines.append(f"{r['dataset']} & {geom} & {method} & {100*r['mean']:.2f} $\\pm$ {100*r['sd']:.2f} & [{100*r['ci95_low']:.2f}, {100*r['ci95_high']:.2f}]" + r' \\')
    lines+=['\\bottomrule','\\end{tabular}']
    (out/'three_dataset_paper_table.tex').write_text('\n'.join(lines)+'\n')
    return rows
