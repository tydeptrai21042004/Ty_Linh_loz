"""Synthetic unit fixtures only: never claim these numbers as trained benchmark results."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import yaml

from paper_research_pipeline import make_plan, model_config, cfg_values, choose_winner, write_plan, DEFAULTS
from paper_robustness import conditions, make_view, aggregate


def test_full_and_reduced_plan_counts_and_no_confirmation(tmp_path):
    full=make_plan(); reduced=make_plan(include_c2former=False)
    assert len(full)==111 and len(reduced)==102
    data=write_plan(full,tmp_path/'experiment_plan.json')
    assert data['counts']=={'total_training':111,'sweep':39,'final':27,'ablation':45}
    assert len({j.id for j in full})==111
    assert set(j.stage for j in full)=={'sweep','final','ablation'}
    for ds in ('vedai','dronevehicle','m3fd'):
        items=[j for j in full if j.dataset==ds]
        assert [sum(j.stage==stage for j in items) for stage in ('sweep','final','ablation')]==[13,9,15]
        assert len(set((j.method,j.seed) for j in items if j.stage=='final'))==9


def test_yaml_overrides_are_real_and_not_paper_baselines(tmp_path):
    base=Path(__file__).parent/'ultralytics/cfg/models/etfnet/etfnet_P2_C3k2_SJPA_TR_Corrected.yaml'
    for j in [x for x in make_plan(['vedai']) if x.stage=='ablation']:
        out=model_config(base,tmp_path/f'{j.id}.yaml',['car','truck'],cfg_values(j),j.ablation)
        content=yaml.safe_load(out.read_text())
        layer=[x for x in content['backbone'] if x[2]=='MR_SJPA'][0]
        assert layer[3][-1]==j.ablation
        assert content['nc']==2
    for j in [x for x in make_plan(['vedai']) if x.stage=='sweep']:
        params=cfg_values(j)
        assert set(params)==set(DEFAULTS)
    assert all('concat' not in j.id and 'goci' not in j.id for j in make_plan())


def test_real_mr_sjpa_ablation_branches_and_gradients():
    import torch
    from ultralytics.nn.modules.block import MR_SJPA
    torch.manual_seed(42)
    rgb=torch.randn(2,8,16,16,requires_grad=True)
    ir=torch.randn(2,8,16,16,requires_grad=True)
    for mode in range(6):
        mod=MR_SJPA(8,groups=2,stat_grid=8,max_shift=1,ablation=mode)
        with torch.no_grad():
            mod.ir_residual_weight.fill_(0.2)
            mod.rgb_residual_weight.fill_(0.2)
        y,diagnostic=mod.forward_with_diagnostics((rgb,ir))
        assert y.shape==(2,16,16,16)
        assert torch.isfinite(y).all()
        assert diagnostic['ablation']==mode
        if mode==1:assert torch.all(diagnostic['offset']==0)
        if mode!=4:
            raw=torch.cat((rgb,ir),1)
            assert torch.all((y-raw).flatten(1).norm(dim=1)<=mod.trust_radius*raw.flatten(1).norm(dim=1)+0.001)
        y.square().mean().backward(retain_graph=True)
    assert torch.isfinite(rgb.grad).all() and torch.isfinite(ir.grad).all()
    with pytest.raises(ValueError):MR_SJPA(8,groups=2,ablation=9)


def test_corruption_conditions_paired_and_not_source_modified(tmp_path):
    root=tmp_path/'source';source=root/'data.yaml'
    for split in ('train','val','test'):
        for target in ('rgb','ir'):
            d=root/target/'images'/split;d.mkdir(parents=True)
            cv2.imwrite(str(d/'a.jpg'),np.full((20,20,3),150 if target=='rgb' else 200,dtype=np.uint8))
        d=root/'labels'/split;d.mkdir(parents=True)
        (d/'a.txt').write_text('0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n')
    data={'path':str(root),'names':['car']}
    for split in ('train','val','test'):
        data[split]=f'rgb/images/{split}';data[split+'_ir']=f'ir/images/{split}';data[split+'_labels']=f'labels/{split}'
    source.write_text(yaml.safe_dump(data))
    assert len(conditions())==21
    orig=(root/'ir/images/test/a.jpg').read_bytes()
    condition=next(x for x in conditions() if x['id']=='dropout_ir')
    output=make_view(source,tmp_path/'views'/'dropout_ir',condition)
    converted=yaml.safe_load(output.read_text())
    img=cv2.imread(str(Path(converted['test_ir'])/'a.png'))
    assert np.max(img)==0
    img=cv2.imread(str(Path(converted['test'])/'a.png'))
    assert np.mean(img)==150
    assert (root/'ir/images/test/a.jpg').read_bytes()==orig
    assert make_view(source,tmp_path/'views'/'dropout_ir',condition)==output


def test_sweep_winner_uses_validation_only(tmp_path):
    jobs=[j for j in make_plan(['vedai']) if j.stage=='sweep']
    for n,j in enumerate(jobs):
        p=tmp_path/j.id;p.mkdir();(p/'training_receipt.json').write_text(json.dumps({'status':'trained','val_best_map50_95':.1+n/100}))
    result=choose_winner('vedai',jobs,tmp_path)
    assert result['winning_job']==jobs[-1].id
    assert result['selection_seed']==0


def test_robustness_aggregate_rejects_fake_incomplete_receipts(tmp_path):
    incomplete={'status':'not_run','eval_split':'test'}
    with pytest.raises(ValueError):aggregate([('clean','mr_sjpa_proposal',incomplete)],tmp_path)
