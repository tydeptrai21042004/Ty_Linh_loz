"""Regression properties for the sole proposed fusion method."""
from pathlib import Path
import copy
import pytest
import torch
from ultralytics.nn.modules.block import MR_SJPA


def make_pair(batch=2, channels=8, spatial=32, shift=(1, -1)):
    torch.manual_seed(20)
    rgb = torch.zeros(batch, channels, spatial, spatial)
    rgb[..., 5:-5, 5:-5] = torch.randn(batch, channels, spatial - 10, spatial - 10)
    ir = MR_SJPA._translate_once(rgb, torch.tensor([shift] * batch))
    return rgb, ir


def test_shift_recovery_and_independent_samples():
    model = MR_SJPA(8, groups=2, stat_grid=16, max_shift=1,
                    shift_penalty=0.0, shift_margin=0.005).eval()
    rgb, ir = make_pair()
    output, diagnostics = model.forward_with_diagnostics([rgb, ir])
    assert diagnostics['offset'].tolist() == [[-1, 1], [-1, 1]]
    assert diagnostics['improvement'].min() > 0
    assert torch.equal(output, torch.cat((rgb, ir), 1))


def test_no_shift_on_identical_pair():
    model = MR_SJPA(8, groups=2, max_shift=1).eval()
    rgb, _ = make_pair()
    _, diagnostics = model.forward_with_diagnostics([rgb, rgb.clone()])
    assert diagnostics['offset'].tolist() == [[0, 0], [0, 0]]


def test_zero_init_trainable_and_bounded():
    model = MR_SJPA(8, groups=2, trust_radius=0.07).train()
    a, b = make_pair()
    a = a.requires_grad_(True)
    b = b.requires_grad_(True)
    out = model([a,b]); z=torch.cat((a,b),1)
    assert torch.equal(out,z)
    out.square().mean().backward()
    assert model.ir_residual_weight.grad is not None
    assert torch.isfinite(model.ir_residual_weight.grad).all()
    assert model.ir_residual_weight.grad.norm() > 0
    with torch.no_grad():
        model.rgb_residual_weight.fill_(1)
        model.ir_residual_weight.fill_(1)
    out, info=model.forward_with_diagnostics([a,b])
    per_sample=(out-z).float().flatten(1).norm(dim=1)
    assert torch.all(per_sample <= 0.07*z.float().flatten(1).norm(dim=1)+1e-4)
    assert torch.isfinite(info['coherence']).all()


def test_batch_independence():
    model=MR_SJPA(8, groups=2).eval()
    rgb,ir=make_pair()
    with torch.no_grad():
        model.ir_residual_weight.fill_(0.2)
    first=model([rgb[:1],ir[:1]])
    combined=model([torch.cat((rgb[:1], 4*rgb[1:2]),0),
                    torch.cat((ir[:1], .5*ir[1:2]),0)])
    assert torch.allclose(first, combined[:1], atol=3e-6, rtol=3e-6)


def test_no_avgpool_resolution_blowup():
    model=MR_SJPA(8, groups=2, stat_grid=8)
    measurements=[]
    for n in (24,48,96):
        torch.manual_seed(77)
        a=torch.randn(2,8,n,n)
        b=a+0.02*torch.randn_like(a)
        r, i, coherence, singular=model._canonical_candidates(a,b)
        measurements.append(r.std().item()/a.std().item())
    assert max(measurements) < 2.0, measurements
    assert max(measurements)/min(measurements) < 1.5, measurements


def test_constant_and_amp_on_cpu():
    model=MR_SJPA(8, groups=2, stat_grid=8)
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        out=model([torch.zeros(2,8,16,16), torch.zeros(2,8,16,16)])
    assert torch.isfinite(out).all()


def test_invalid_shapes_and_parameters():
    with pytest.raises(ValueError): MR_SJPA(8,groups=2,max_shift=-1)
    m=MR_SJPA(8,groups=2)
    with pytest.raises(ValueError): m([torch.zeros(1,8,10,10),torch.zeros(1,8,9,10)])


def test_unsupported_export_fails_closed():
    with pytest.raises(RuntimeError, match='not certified'):
        MR_SJPA(8,groups=2).prepare_for_export()


def test_paper_only_registry_and_kaggle_matrix():
    import json
    root=Path(__file__).resolve().parents[1]
    registry=json.loads((root/'PAPER_BASELINE_PROVENANCE.json').read_text())
    assert registry['local_dummy_baselines']==[]
    assert {x['name'] for x in registry['paper_baselines']}=={'ETFNet','C2Former'}
    runner=(root/'paper_benchmark_runner.py').read_text()
    assert 'def official_source' in runner
    assert 'etfnet_official_paper' in runner
    for dummy in ('dual_add','cafem_only','goci_no_spatial','rtpf_safety_ablation'):
        assert dummy not in runner
